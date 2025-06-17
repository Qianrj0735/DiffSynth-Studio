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
 CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" python examples/wanvideo/train_wan_future_fmpk.py \
 --task train \
 --train_architecture full \
 --dataset_path "epic" \
 --output_path ./log \
 --dit_path /root/.cache/huggingface/hub/models--alibaba-pai--Wan2.1-Fun-1.3B-InP/snapshots/7693adbe4ae670234958e42aa98145b4a406116c/diffusion_pytorch_model.safetensors \
 --image_encoder_path /root/.cache/huggingface/hub/models--alibaba-pai--Wan2.1-Fun-1.3B-InP/snapshots/7693adbe4ae670234958e42aa98145b4a406116c/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth \
 --vae_path "/root/.cache/huggingface/hub/models--alibaba-pai--Wan2.1-Fun-1.3B-InP/snapshots/7693adbe4ae670234958e42aa98145b4a406116c/Wan2.1_VAE.pth" \
 --steps_per_epoch 10000 \
 --max_epochs 100 \
 --learning_rate 1e-4 \
 --accumulate_grad_batches 4 \
 --use_gradient_checkpointing \
 --num_frames 193 \
 --dataloader_num_workers 25 \
 --s_path epic_s \
 --xs_path epic_s \
 --ms_path epic_m_interval_4\
 --predict_config /home/workspace/DiffSynth-Studio/examples/wanvideo/predict_pack_configs/6m12ms.yml \
 --predicting_indice "0,1,2,3,4,5" \
 --packed_indice "6,7,8,9,10,11,12,13,14,15,16,17" \
 --num_validation_blocks 8 \
 --num_val_batches 2 \
 --is_absolute_rope
