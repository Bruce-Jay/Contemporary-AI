# !/usr/bin/env bash

# 只图像
PYTHONIOENCODING=utf-8 CUDA_LAUNCH_BLOCKING=1 python run.py --num_train_epochs 10 --output_dir ./output/ --do_train 1 --do_eval 1 --crop_size 80 --image_only 1 --do_test 0 --train_file train_legacy.tsv --valid_file valid_legacy.tsv