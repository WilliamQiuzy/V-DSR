#!/bin/bash
set -x
export GPU=$(nvidia-smi --list-gpus | wc -l)
torchrun --nproc-per-node=$GPU run.py --data Video-MME --model Qwen2.5-VL-7B-Instruct-ForVideo-mine --work-dir ./50k_test_all