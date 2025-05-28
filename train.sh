# CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" python examples/wanvideo/train_wan_framepack_i2v.py \
#  --task train \
#  --train_architecture full \
#  --dataset_path "385" \
#  --output_path ./log \
#  --dit_path /root/.cache/huggingface/hub/models--alibaba-pai--Wan2.1-Fun-1.3B-InP/snapshots/7693adbe4ae670234958e42aa98145b4a406116c/diffusion_pytorch_model.safetensors \
#  --image_encoder_path /root/.cache/huggingface/hub/models--alibaba-pai--Wan2.1-Fun-1.3B-InP/snapshots/7693adbe4ae670234958e42aa98145b4a406116c/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth \
#  --vae_path /root/.cache/huggingface/hub/models--alibaba-pai--Wan2.1-Fun-1.3B-InP/snapshots/7693adbe4ae670234958e42aa98145b4a406116c/Wan2.1_VAE.pth \
#  --steps_per_epoch 10000 \
#  --max_epochs 20 \
#  --learning_rate 1e-4 \
#  --accumulate_grad_batches 8 \
#  --use_gradient_checkpointing \
#  --num_frames 385 \
#  --dataloader_num_workers 25

#  CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" python examples/wanvideo/train_wan_framepack_i2v.py \
#  --task train \
#  --train_architecture full \
#  --dataset_path "385" \
#  --output_path ./log \
#  --dit_path /root/.cache/huggingface/hub/models--alibaba-pai--Wan2.1-Fun-1.3B-InP/snapshots/7693adbe4ae670234958e42aa98145b4a406116c/diffusion_pytorch_model.safetensors \
#  --image_encoder_path /root/.cache/huggingface/hub/models--alibaba-pai--Wan2.1-Fun-1.3B-InP/snapshots/7693adbe4ae670234958e42aa98145b4a406116c/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth \
#  --vae_path /root/.cache/huggingface/hub/models--alibaba-pai--Wan2.1-Fun-1.3B-InP/snapshots/7693adbe4ae670234958e42aa98145b4a406116c/Wan2.1_VAE.pth \
#  --steps_per_epoch 10000 \
#  --max_epochs 20 \
#  --learning_rate 1e-5 \
#  --accumulate_grad_batches 8 \
#  --use_gradient_checkpointing \
#  --num_frames 385 \
#  --dataloader_num_workers 25
 CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" python examples/wanvideo/train_wan_framepack_i2v.py \
 --task train \
 --train_architecture full \
 --dataset_path "/workspace/DiffSynth-Studio/epic" \
 --output_path ./log \
 --dit_path /root/.cache/huggingface/hub/models--alibaba-pai--Wan2.1-Fun-1.3B-InP/snapshots/7693adbe4ae670234958e42aa98145b4a406116c/diffusion_pytorch_model.safetensors \
 --image_encoder_path /root/.cache/huggingface/hub/models--alibaba-pai--Wan2.1-Fun-1.3B-InP/snapshots/7693adbe4ae670234958e42aa98145b4a406116c/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth \
 --vae_path /root/.cache/huggingface/hub/models--alibaba-pai--Wan2.1-Fun-1.3B-InP/snapshots/7693adbe4ae670234958e42aa98145b4a406116c/Wan2.1_VAE.pth \
 --steps_per_epoch 10000 \
 --max_epochs 100 \
 --learning_rate 1e-5 \
 --accumulate_grad_batches 8 \
 --use_gradient_checkpointing \
 --num_frames 193 \
 --dataloader_num_workers 25