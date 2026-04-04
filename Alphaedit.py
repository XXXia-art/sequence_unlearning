#!/usr/bin/env python3
import os, re, time
import argparse
import warnings
warnings.filterwarnings("ignore")

import torch
import pandas as pd
from tqdm import tqdm
from kmeans_pytorch import kmeans
from diffusers import StableDiffusionPipeline

from src.utils import seed_everything

def get_token_id(prompt, tokenizer, return_ids_only=True):
    tokens = tokenizer(
        prompt,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt"
    )
    return tokens.input_ids if return_ids_only else tokens

## AlphaEdit
@torch.no_grad()
def edit_model(
    args,
    pipeline,
    base_state,
    all_target,
    target_concepts,
    hist_targets,
    anchor_concepts,
    retain_texts,
    emb_size=768,
    chunk_size=128,
    device="cuda"
):
    I = torch.eye(emb_size, device=device)

    ## choose params
    if args.params == "V":
        edit_dict = { k: v.clone() for k, v in pipeline.unet.state_dict().items() if "attn2.to_v" in k }
    elif args.params == "K":
        edit_dict = { k: v.clone() for k, v in pipeline.unet.state_dict().items() if "attn2.to_k" in k }
    elif args.params == "KV":
        edit_dict = { k: v.clone() for k, v in pipeline.unet.state_dict().items() if "attn2.to_k" in k or "attn2.to_v" in k}
    else:
        raise ValueError("Invalid --params")

    # ---- null cluster ----
    null_inputs = get_token_id("", pipeline.tokenizer, return_ids_only=False)
    null_hidden = pipeline.text_encoder(null_inputs.input_ids.to(device)).last_hidden_state[0]
    _, centers = kmeans(X=null_hidden[1:], num_clusters=3, device=device)
    K2 = torch.cat([null_hidden[[0]], centers.to(device)], dim=0).T
    I2 = torch.eye(K2.shape[1], device=device)

    ## Target / Anchor
    sum_tt, sum_at, ke, hist_ke = [], [], [], []

    for t, a in zip(target_concepts, anchor_concepts):
        t_in = get_token_id(t, pipeline.tokenizer, return_ids_only=False)
        a_in = get_token_id(a, pipeline.tokenizer, return_ids_only=False)

        t_emb = pipeline.text_encoder(t_in.input_ids.to(device)).last_hidden_state[0]
        a_emb = pipeline.text_encoder(a_in.input_ids.to(device)).last_hidden_state[0]

        idx_t = t_in.attention_mask[0].sum().item() - 2
        idx_a = a_in.attention_mask[0].sum().item() - 2

        t_vec = t_emb[[idx_t]]
        a_vec = a_emb[[idx_a]]

        sum_tt.append(t_vec.T @ t_vec)
        sum_at.append(a_vec.T @ t_vec)
        ke.append(t_vec.T)
    sum_tt = torch.stack(sum_tt).mean(0)
    sum_at = torch.stack(sum_at).mean(0)
    # k_e = torch.cat([x for x in ke], dim=1)   # [d, n_target]
    k_e = torch.stack(ke).mean(0) 

    ## hist_kp计算
    for h in hist_targets:
        h_in = get_token_id(h, pipeline.tokenizer, return_ids_only=False)
        h_emb = pipeline.text_encoder(h_in.input_ids.to(device)).last_hidden_state[0]
        idx_h = h_in.attention_mask[0].sum().item() - 2
        h_vec = h_emb[[idx_h]]   
        hist_ke.append(h_vec.T @ h_vec)  

    if len(hist_ke) > 0:
        sum_hh = torch.stack(hist_ke).mean(0)
    else:
        sum_hh = torch.zeros_like(I)

    ## Retain
    retain_texts = [x for x in retain_texts if not any(c.lower() in x.lower() for c in all_target)]
    last_ret_embs = []

    for i in range(0, len(retain_texts), chunk_size):
        r_in = get_token_id(retain_texts[i:i + chunk_size], pipeline.tokenizer, return_ids_only=False)
        r_emb = pipeline.text_encoder(r_in.input_ids.to(device)).last_hidden_state
        idx = r_in.attention_mask.sum(1) - 2
        last_ret_embs.append( r_emb[torch.arange(r_emb.size(0)), idx].unsqueeze(1))

    last_ret_embs = torch.cat(last_ret_embs, dim=0)
    last_ret_embs = last_ret_embs[torch.randperm(last_ret_embs.size(0))] ## shuffle

    ## 记录指标
    match = re.search(r"step_(\d+)", args.save_path)
    step = int(match.group(1))
    config_path = os.path.join(os.path.dirname(os.path.dirname(args.save_path)), "config.txt")
    with open(config_path, "a") as f:
                f.write(
                    f"[step {step:03d}] >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> \n" )

    global_hist_norm_sq = 0.0
    global_cur_norm_sq  = 0.0

    ## Edit each layer
    for name, W in tqdm(edit_dict.items(), desc="Editing"):

        W = W.to(device)
        W_orig = base_state[name].to(device)
        delta_history = W - W_orig  # 历史累计修改

        resp = delta_history @ k_e          
        noise = resp.norm(dim=0).pow(2)
        with open(config_path, "a") as f:
            f.write(
                f"layer={name}| \n"
                f"noise={float(noise):.8f} | "
            )
        with open(config_path, "a") as f:
                f.write(f"\n")

        layer_ret_embs = last_ret_embs  # 使用所有保留样本，不做IPF筛选
        sum_rr, n = [], 0
        for i in range(0, len(layer_ret_embs), chunk_size):
            chunk = layer_ret_embs[i:i + chunk_size]
            n += chunk.size(0)
            sum_rr.append((chunk.transpose(1, 2) @ chunk).sum(0))
        sum_rr = torch.stack(sum_rr).sum(0) / max(1, n)

        U, S, _ = torch.svd(sum_rr)
        mask = S < args.threshold
        if mask.sum() == 0:
            continue

        P = U[:, mask] @ U[:, mask].T
        M = (sum_tt @ P + sum_hh @ P + args.retain_scale * I).inverse()
        delta = (
            W @ (sum_at - sum_tt) @ P
            @ (I - M @ K2 @ (K2.T @ P @ M @ K2 + args.lamb * I2).inverse() @ K2.T @ P)
            @ M
        )

        delta_total = delta_history + delta
        global_hist_norm_sq += torch.norm(delta_total @ k_e) ** 2
        global_cur_norm_sq += torch.norm(delta @ k_e) ** 2
        edit_dict[name] = (W + delta).cpu()

    noise_e = abs(global_hist_norm_sq - global_cur_norm_sq)
    return edit_dict, noise_e


# -------------------------------------------------
# Main
# -------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--sd_ckpt', help='base version for stable diffusion', type=str, default='/home/lzh/xpz/model_weight/SD/models--CompVis--stable-diffusion-v1-4/snapshots/133a221b8aa7292a167afc5127cb63fb5005638b')
    parser.add_argument("--edit_ckpt", help="save history weight,m,v", type=str, default=None)
    parser.add_argument("--save_path", type=str, required=True)

    parser.add_argument("--target_concepts", type=str, required=True)
    parser.add_argument("--anchor_concepts", type=str, required=True)
    parser.add_argument("--retain_path", type=str, default=None)
    parser.add_argument("--heads", type=str, default="concept")

    parser.add_argument("--params", type=str, default="V")
    parser.add_argument("--aug_num", type=int, default=10)
    parser.add_argument("--threshold", type=float, default=1e-1)
    parser.add_argument("--retain_scale", type=float, default=1.0)
    parser.add_argument("--lamb", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dtype", type=str, default="float32")

    args = parser.parse_args()

    device = "cuda"
    seed_everything(args.seed)

    dtype = torch.float16 if args.dtype == "float16" else torch.float32

    # ---- Load pipeline ----
    pipe = StableDiffusionPipeline.from_pretrained(args.sd_ckpt, torch_dtype=dtype,)
    pipe.safety_checker = None
    pipe.feature_extractor = None
    pipe.vae = None   # ⭐ 关键，edit 完全不需要 VAE

    pipe.text_encoder.to(device)
    pipe.unet.to(device)

    basestate = {k: v.clone() for k, v in pipe.unet.state_dict().items()}

    # ----  SEQUENTIAL LOAD POINT ---- ## 顺序编辑实现
    if args.edit_ckpt and os.path.exists(args.edit_ckpt):
        print(f"[INFO] Loading previous edit: {args.edit_ckpt}")
        prev = torch.load(args.edit_ckpt, map_location="cpu")
        pipe.unet.load_state_dict(prev, strict=False)

    # ---- Parse inputs ----
    all_targets = [x.strip() for x in args.target_concepts.split(",")]
    current_target = all_targets[-5:]
    hist_targets = all_targets[:-len(current_target)]

    anchors = [x.strip() for x in args.anchor_concepts.split(",")]
    if len(anchors) == 1:
        anchors = anchors * len(current_target)

    retain_texts = [""]
    if args.retain_path:
        df = pd.read_csv(args.retain_path)
        retain_texts = df[args.heads].unique().tolist()

    torch.cuda.empty_cache()
    # ---- Edit ----
    edit_dict, noise_e = edit_model(
        args, pipe, basestate, all_targets, current_target, hist_targets, anchors, retain_texts, device=device
    )

    os.makedirs(args.save_path, exist_ok=True)
    save_file = os.path.join(args.save_path, "weight.pt")
    torch.save(edit_dict, save_file)
    print(f"[DONE] Saved {save_file}")
    # ---- noise_E log path ----
    log_dir = os.path.dirname(os.path.dirname(args.save_path))
    match = re.search(r"step_(\d+)", args.save_path)
    step = int(match.group(1))
    os.makedirs(log_dir, exist_ok=True)
    txt_path = os.path.join(log_dir, "noise_e_log.txt")
    with open(txt_path, "a") as f:
        f.write(f"step: {step}  noise_e: {noise_e:.8f}\n")

    values = []
    with open(txt_path, "r") as f:
        for line in f:
            if "noise_e:" in line:
                val = float(line.strip().split("noise_e:")[1])
                values.append(val)
    noise_E = sum(values) / len(values)
    noise_E_path = os.path.join(log_dir, "noise_E_log.txt")
    with open(noise_E_path, "a") as f:
        f.write(f"step: {step}  noise_E: {noise_E:.8f}\n")
    print(f"[INFO] noise_E updated: {noise_E:.8f}")