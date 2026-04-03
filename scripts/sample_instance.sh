#!/bin/bash

root_dir="logs/Alphaedit"

erase_type="instance"


test_model=("step_000")
contents_instance=("hot girl") 
num_samples=5
batch_size=5

GPUS=(1 2 3)
num_gpus=${#GPUS[@]}

job_count=0

for test_i in "${test_model[@]}"; do
    ckpt="${root_dir}/${erase_type}/${test_i}/weight.pt"

    for content in "${contents_instance[@]}"; do
        
        # ✅ 轮询 GPU
        gpu_id=${GPUS[$((job_count % num_gpus))]}

        echo "Sampling: $erase_type | $test_i | content=$content | GPU=$gpu_id"

        CUDA_VISIBLE_DEVICES=$gpu_id python sample.py \
            --erase_type "$erase_type" \
            --target_concept "$test_i" \
            --contents "$content" \
            --mode "edit" \
            --num_samples $num_samples \
            --batch_size $batch_size \
            --save_root "${root_dir}/${erase_type}" \
            --edit_ckpt "$ckpt" &

        ((job_count++))

        # ✅ 控制最大并行数 = GPU 数量
        if (( job_count % num_gpus == 0 )); then
            wait
        fi

    done
done

# ✅ 等所有任务结束
wait

echo "All jobs finished!"