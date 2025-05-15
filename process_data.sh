# /home/io/.cache/huggingface/hub/models--alibaba-pai--Wan2.1-Fun-1.3B-InP/snapshots/7693adbe4ae670234958e42aa98145b4a406116c

CUDA_VISIBLE_DEVICES="0" python examples/wanvideo/train_wan_t2v.py \
  --task data_process \
  --dataset_path panda_100_pick_red_cube_randcam \
  --output_path panda_100_pick_red_cube_randcam/processed \
  --text_encoder_path "/home/io/.cache/huggingface/hub/models--alibaba-pai--Wan2.1-Fun-1.3B-InP/snapshots/7693adbe4ae670234958e42aa98145b4a406116c/models_t5_umt5-xxl-enc-bf16.pth"\
  --vae_path "/home/io/.cache/huggingface/hub/models--alibaba-pai--Wan2.1-Fun-1.3B-InP/snapshots/7693adbe4ae670234958e42aa98145b4a406116c/Wan2.1_VAE.pth"\
  --image_encoder_path "/home/io/.cache/huggingface/hub/models--alibaba-pai--Wan2.1-Fun-1.3B-InP/snapshots/7693adbe4ae670234958e42aa98145b4a406116c/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"\
  --tiled \
  --num_frames 97 \
  --height 256 \
  --width 256