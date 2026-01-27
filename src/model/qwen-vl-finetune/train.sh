#!/bin/bash
# Complete QwenVL Training Launch Script with Full Parameter Documentation

# ======================
# Distributed Configuration
# ======================
MASTER_ADDR="127.0.0.1"                     # [Required] Master node IP for multi-GPU training
MASTER_PORT=$(shuf -i 20000-29999 -n 1)     # Random port to avoid conflicts
NPROC_PER_NODE=$(nvidia-smi --list-gpus | wc -l)  # Automatically detects available GPUs
#NPROC_PER_NODE=8
# ======================
# Path Configuration
# ======================
MODEL_PATH="Qwen/Qwen2.5-VL-7B-Instruct"  # [ModelArguments] Pretrained model path
OUTPUT_DIR="./spatial_reasoning"                   # Directory for saving checkpoints
CACHE_DIR="./cache"                          # [TrainingArguments] Cache directory for models

# ======================
# Model Configuration
# ======================
DATASETS="spatial_reasoning%100"                  # [DataArguments] Dataset with sampling rate

# ======================
# Training Hyperparameters
# ======================
torchrun --nproc_per_node=$NPROC_PER_NODE \
         --master_addr=$MASTER_ADDR \
         --master_port=$MASTER_PORT \
         qwenvl/train/train_qwen.py \
         --model_name_or_path $MODEL_PATH \
         --tune_mm_llm True \
         --tune_mm_vision False \
         --tune_mm_mlp True \
         --pi3_path PATH_TO_Pi3 \
         --dataset_use $DATASETS \
         --output_dir $OUTPUT_DIR \
         --cache_dir $CACHE_DIR \
         --bf16 \
         --per_device_train_batch_size 1 \
         --gradient_accumulation_steps 4 \
         --learning_rate 2e-7 \
         --mm_projector_lr 1e-6 \
         --vision_tower_lr 1e-6 \
         --optim adamw_torch \
         --model_max_length 8192 \
         --data_flatten True \
         --data_packing True \
         --max_pixels 230400 \
         --min_pixels 784 \
         --base_interval 1 \
         --video_max_frames 32 \
         --video_min_frames 32 \
         --video_max_frame_pixels 230400 \
         --video_min_frame_pixels 784 \
         --num_train_epochs 1 \
         --warmup_ratio 0.03 \
         --lr_scheduler_type "cosine" \
         --weight_decay 0.01 \
         --logging_steps 10 \
         --save_steps 100 \
         --save_total_limit 3 \
         --max_grad_norm 1 \
         --report_to tensorboard \
         --deepspeed ./scripts/zero3_offload.json \