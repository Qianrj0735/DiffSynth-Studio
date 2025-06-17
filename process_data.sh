# /root/.cache/huggingface/hub/models--alibaba-pai--Wan2.1-Fun-1.3B-InP/snapshots/7693adbe4ae670234958e42aa98145b4a406116c

CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" python examples/wanvideo/train_wan_t2v.py \
  --task data_process \
  --dataset_path "/home/workspace/DiffSynth-Studio/epic" \
  --output_path "/home/workspace/DiffSynth-Studio/epic/processed" \
  --text_encoder_path "/root/.cache/huggingface/hub/models--alibaba-pai--Wan2.1-Fun-1.3B-InP/snapshots/7693adbe4ae670234958e42aa98145b4a406116c/models_t5_umt5-xxl-enc-bf16.pth"\
  --vae_path "/root/.cache/huggingface/hub/models--alibaba-pai--Wan2.1-Fun-1.3B-InP/snapshots/7693adbe4ae670234958e42aa98145b4a406116c/Wan2.1_VAE.pth"\
  --image_encoder_path "/root/.cache/huggingface/hub/models--alibaba-pai--Wan2.1-Fun-1.3B-InP/snapshots/7693adbe4ae670234958e42aa98145b4a406116c/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"\
  --tiled \
  --num_frames 193 \
  --height 576 \
  --width 1024 \
  --frame_interval 4 \
  --save_pth_path "epic_m_interval_4"