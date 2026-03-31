#!/bin/bash

baseline="SPEED"
save_root="logs/Alphaedit"
erase_type="instance"


CUDA_VISIBLE_DEVICES=1 python src/clip_score_cal.py \
  --contents "coco" \
  --root_path "${save_root}/${erase_type}"
