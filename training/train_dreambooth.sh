export MODEL_NAME="../share/runwayml/stable-diffusion-v1-5"
# export INSTANCE_DIR="dog"
export OUTPUT_DIR_root="dreambooth_model_ori"
export CLASS_DIR_root="dreambooth_class-images"
export DATA_SET_root="./dreambooth/dataset"

export subject_name=("backpack_dog" "backpack" "bear_plushie" "berry_bowl" "can" "candle" "cat" "cat2" "clock" "colorful_sneaker" "dog" "dog2" "dog3" "dog5" "dog6" "dog7" "dog8" "duck_toy" "fancy_boot" "grey_sloth_plushie" "monster_toy" "pink_sunglasses" "poop_emoji" "rc_car" "red_cartoon" "robot_toy" "shiny_sneaker" "teapot" "vase" "wolf_plushie")
export class_name=("backpack" "backpack" "stuffed animal" "bowl" "can" "candle" "cat" "cat" "clock" "sneaker" "dog" "dog" "dog" "dog" "dog" "dog" "dog" "toy" "boot" "stuffed animal" "toy" "glasses" "toy" "toy" "cartoon" "toy" "sneaker" "teapot" "vase" "stuffed animal")

for i in "${!subject_name[@]}"; do
    export DATA_SET="$DATA_SET_root/${subject_name[i]}"  # Append each subject to the dataset path
    class="${class_name[i]}"
    export CLASS_DIR="$CLASS_DIR_root/$class"
    export OUTPUT_DIR="$OUTPUT_DIR_root/${subject_name[i]}"
    # echo "DataSet: $DATA_SET, class: $class, subject: ${subject_name[i]}, class_dir: $CLASS_DIR, output_dir: $OUTPUT_DIR"
    accelerate launch --main_process_port 29501 --mixed_precision="fp16" --multi_gpu --num_processes 6 --gpu_ids='all' train_dreambooth.py \
    --pretrained_model_name_or_path=$MODEL_NAME  \
    --instance_data_dir=$DATA_SET \
    --class_data_dir="$CLASS_DIR" \
    --output_dir="$OUTPUT_DIR" \
    --with_prior_preservation --prior_loss_weight=1.0 \
    --instance_prompt="a photo of sks ${subject_name[i]}" \
    --class_prompt="a photo of $class" \
    --resolution=512 \
    --train_batch_size=1 \
    --gradient_accumulation_steps=1 \
    --learning_rate=5e-6 \
    --lr_scheduler="constant" \
    --lr_warmup_steps=0 \
    --num_class_images=200 \
    --max_train_steps=800 \
  #   --push_to_hub
done

