# !/usr/bin/env bash

# 图像+文字
PYTHONIOENCODING=utf-8 CUDA_LAUNCH_BLOCKING=1 python run.py --output_dir ./output/ --do_train 1 --do_eval 1 --num_train_epochs 10 --warmup_proportion 0 --do_test 0 --train_file train_legacy.tsv --valid_file valid_legacy.tsv