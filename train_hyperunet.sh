export MODEL_NAME="../share/runwayml/stable-diffusion-v1-5"
export DATASET_NAME="../fantasyfish/laion-art"
#   --use_ema \ 
#-
accelerate launch --main_process_port 29501 --mixed_precision="fp16" --multi_gpu --num_processes 8 --gpu_ids='all' train_hyperunet.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --dataset_name=$DATASET_NAME \
  --image_column="image" \
  --caption_column="text" \
  --resolution=512 --center_crop --random_flip \
  --train_batch_size=4 \
  --gradient_accumulation_steps=1 \
  --gradient_checkpointing \
  --num_train_epochs 15 \
  --checkpointing_steps=500 \
  --learning_rate=1e-05 \
  --max_grad_norm=1 \
  --use_8bit_adam \
  --lr_scheduler="constant" --lr_warmup_steps=0 \
  --seed=42 \
  --enable_xformers_memory_efficient_attention \
  --output_dir="sd-naruto-model_hard_improve1_factor4_trans_decoder_sd1.5_factorablation_epoch15"

  #  # --max_train_steps=5000 \