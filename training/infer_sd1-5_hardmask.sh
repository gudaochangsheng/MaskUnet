torchrun --master-port=29506 --nproc_per_node=8 infer_sd1-5_hardmask.py \
  --pretrained_model_name_or_path "PixArt-alpha/PixArt-XL-2-512x512" \
  --captions_file "captions_2017.txt" \
  --world_size 8 \
  --batch_size 16