# !/usr/bin/env bash

PYTHONIOENCODING=utf-8 CUDA_LAUNCH_BLOCKING=1 python run.py --output_dir ./output/ --do_train 0 --do_eval 0 --crop_size 80 --do_test 1 --train_file train_legacy.tsv --valid_file valid_legacy.tsv