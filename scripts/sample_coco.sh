#!/bin/bash

save_root="logs/Alphaedit"
erase_type="instance"

# ============================
# 🔥 在这里自定义你要采样的 step
# ============================
STEPS=(00 10 20 30 50 60 70 80 90)


EVAL_GPUS=(0 1 2)
NUM_EVAL_GPUS=${#EVAL_GPUS[@]}
EVAL_BATCH_SIZE=40


for ((i=0; i<${#STEPS[@]}; i++)); do
  
  step_id=${STEPS[$i]}
  step_name=$(printf "step_%03d" "$step_id")
  ckpt="${save_root}/${erase_type}/${step_name}/weight.pt"

  # 如果该 step 没训练完，直接跳过
  if [ ! -f "$ckpt" ]; then
    echo "[SKIP] ${step_name} not found"
    continue
  fi

  gpu=${EVAL_GPUS[$(( i % NUM_EVAL_GPUS ))]}

  echo "[SAMPLE] step=${step_name}  GPU=${gpu}"

  CUDA_VISIBLE_DEVICES=$gpu python sample2.py \
    --erase_type "instance" \
    --contents "coco" \
    --mode "edit" \
    --num_samples 1 \
    --batch_size "$EVAL_BATCH_SIZE" \
    --save_root "${save_root}/${erase_type}/${step_name}" \
    --edit_ckpt "$ckpt" &

  # 控制并行数 = GPU 数量
  if (( (i + 1) % NUM_EVAL_GPUS == 0 )); then
    wait
  fi

done

wait
echo "✅ Sampling finished"
