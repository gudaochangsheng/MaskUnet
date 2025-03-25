export MODEL_NAME="../share/runwayml/stable-diffusion-v1-5"
export INSTANCE_DIR="./dreambooth/dataset/dog2"
export CLASS_DIR="dream_booth_class_image"
export OUTPUT_DIR="model_path_subject_dream_booth_dog"

accelerate launch --main_process_port 29501 --mixed_precision="fp16" --multi_gpu --num_processes 6 --gpu_ids='all' train_dreambooth_maskunet.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --train_batch_size=4 \
  --gradient_accumulation_steps=1 \
  --learning_rate=1e-3 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --instance_prompt="a photo of sks dog" \
  --resolution=512 \
  --max_train_steps=5000
    # --class_data_dir=$CLASS_DIR \
  # --with_prior_preservation --prior_loss_weight=1.0 \
  
  # --class_prompt="a photo of backpack" \
  
  
  # --num_class_images=200 \
  
  # --push_to_hub