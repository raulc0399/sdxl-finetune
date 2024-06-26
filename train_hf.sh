#!/bin/bash

export MODEL_NAME="stabilityai/stable-diffusion-xl-base-1.0"
export VAE_NAME="madebyollin/sdxl-vae-fp16-fix"
export DATASET_DIR="../PixArt-alpha-finetuning/data/lego-city-adventures-captions"
export CAPTION_COLUMN="llava_caption_with_orig_caption"

accelerate launch train_text_to_image_lora_sdxl.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --pretrained_vae_model_name_or_path=$VAE_NAME \
  --train_data_dir=$DATASET_DIR --caption_column=$CAPTION_COLUMN \
  --resolution=1024 --center_crop \
  --train_batch_size=2 \
  --gradient_accumulation_steps=1 --gradient_checkpointing \
  --num_train_epochs=20 \
  --learning_rate=3e-04 --lr_scheduler="constant" --lr_warmup_steps=0 \
  --mixed_precision="fp16" \
  --report_to="wandb" \
  --rank=8 \
  --validation_prompt="Image in lego city adventures style, cute dragon creature" --validation_epochs=2 \
  --checkpointing_steps=200 \
  --output_dir="sdxl-lego-city-model" \
  --seed=42 \
  --adam_weight_decay=0.03 --adam_epsilon=1e-10 \
  --dataloader_num_workers=12 \
  --snr_gamma=5 

# --learning_rate=3e-04 --lr_scheduler="cosine" --lr_warmup_steps=0 \
