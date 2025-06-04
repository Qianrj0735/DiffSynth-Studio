# train_wan_t2v_accelerate.py
import os
import torch
import argparse
import pandas as pd
import numpy as np
import imageio
from PIL import Image
from torchvision.transforms import v2
from einops import rearrange
from accelerate import Accelerator  # <--- 新增
from peft import LoraConfig, inject_adapter_in_model

torch.manual_seed(42)

from diffsynth import WanVideoPipeline, ModelManager, load_state_dict


class TextVideoDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        base_path,
        metadata_path,
        max_num_frames=81,
        frame_interval=1,
        num_frames=81,
        height=480,
        width=832,
        is_i2v=False,
    ):
        metadata = pd.read_csv(metadata_path)
        self.path = [
            os.path.join(base_path, "train", file_name)
            for file_name in metadata["file_name"]
        ]
        self.text = metadata["text"].to_list()
        self.max_num_frames = max_num_frames
        self.frame_interval = frame_interval
        self.num_frames = num_frames
        self.height = height
        self.width = width
        self.is_i2v = is_i2v

        self.frame_process = v2.Compose(
            [
                v2.CenterCrop(size=(height, width)),
                v2.Resize(size=(height, width), antialias=True),
                v2.ToTensor(),
                v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

    def crop_and_resize(self, image):
        width, height = image.size
        scale = max(self.width / width, self.height / height)
        image = torchvision.transforms.functional.resize(
            image,
            (round(height * scale), round(width * scale)),
            interpolation=torchvision.transforms.InterpolationMode.BILINEAR,
        )
        return image

    def load_frames_using_imageio(
        self,
        file_path,
        max_num_frames,
        start_frame_id,
        interval,
        num_frames,
        frame_process,
    ):
        reader = imageio.get_reader(file_path)
        total = reader.count_frames()
        if (
            total < max_num_frames
            or total - 1 < start_frame_id + (num_frames - 1) * interval
        ):
            reader.close()
            return None

        frames = []
        first_frame = None
        for frame_id in range(num_frames):
            frame = reader.get_data(start_frame_id + frame_id * interval)
            frame = Image.fromarray(frame)
            frame = self.crop_and_resize(frame)
            if first_frame is None:
                first_frame = frame
            frame = frame_process(frame)
            frames.append(frame)
        reader.close()

        frames = torch.stack(frames, dim=0)
        frames = rearrange(frames, "T C H W -> C T H W")

        first_frame = v2.functional.center_crop(
            first_frame, output_size=(self.height, self.width)
        )
        first_frame = np.array(first_frame)

        if self.is_i2v:
            return frames, first_frame
        else:
            return frames

    def load_video(self, file_path):
        start_frame_id = torch.randint(
            0, self.max_num_frames - (self.num_frames - 1) * self.frame_interval, (1,)
        )[0]
        frames = self.load_frames_using_imageio(
            file_path,
            self.max_num_frames,
            start_frame_id,
            self.frame_interval,
            self.num_frames,
            self.frame_process,
        )
        return frames

    def is_image(self, file_path):
        ext = file_path.split(".")[-1].lower()
        return ext in ["jpg", "jpeg", "png", "webp"]

    def load_image(self, file_path):
        frame = Image.open(file_path).convert("RGB")
        frame = self.crop_and_resize(frame)
        first_frame = frame
        frame = self.frame_process(frame)
        frame = rearrange(frame, "C H W -> C 1 H W")
        return frame

    def __getitem__(self, data_id):
        text = self.text[data_id]
        path = self.path[data_id]
        if self.is_image(path):
            if self.is_i2v:
                raise ValueError(f"{path} 不是视频，I2V 模型不支持图像到图像的训练。")
            video = self.load_image(path)
        else:
            video = self.load_video(path)
        if self.is_i2v:
            video, first_frame = video
            return {
                "text": text,
                "video": video,
                "path": path,
                "first_frame": first_frame,
            }
        else:
            return {"text": text, "video": video, "path": path}

    def __len__(self):
        return len(self.path)


class TensorDataset(torch.utils.data.Dataset):
    def __init__(self, base_path, metadata_path, steps_per_epoch, num_frames=193):
        metadata = pd.read_csv(metadata_path)
        self.path = [
            os.path.join(base_path, "train", file_name)
            for file_name in metadata["file_name"]
        ]
        print(len(self.path), "个视频在 metadata 中。")
        self.path = [
            i + ".tensors.pth" for i in self.path if os.path.exists(i + ".tensors.pth")
        ]
        print(len(self.path), "个张量已缓存在 metadata 中。")
        assert len(self.path) > 0, "未找到任何 .tensors.pth 文件。"
        self.num_frames = num_frames
        self.steps_per_epoch = steps_per_epoch

    def __getitem__(self, index):
        data_id = torch.randint(0, len(self.path), (1,))[0]
        data_id = (data_id + index) % len(self.path)  # 保持随机性同时兼容固定 seed
        path = self.path[data_id]
        clip_idx = ((self.num_frames - 1) // 4) + 1
        data = torch.load(path, map_location="cpu")
        data["latents"] = data["latents"][:, :clip_idx]
        data["image_emb"]["y"] = data["image_emb"]["y"][:, :, :clip_idx]
        return data

    def __len__(self):
        return self.steps_per_epoch


class WanTrainer:
    def __init__(self, args):
        self.args = args
        # 1) 初始化 ModelManager 并加载 DiT
        model_manager = ModelManager(torch_dtype=torch.bfloat16, device="cpu")
        if os.path.isfile(args.dit_path):
            model_manager.load_models([args.dit_path])
        else:
            dit_paths = args.dit_path.split(",")
            model_manager.load_models(dit_paths)
        self.pipe = WanVideoPipeline.from_model_manager(model_manager)
        self.pipe.scheduler.set_timesteps(1000, training=True)
        self.freeze_parameters()
        if args.train_architecture == "lora":
            self.add_lora_to_model(
                self.pipe.denoising_model(),
                lora_rank=args.lora_rank,
                lora_alpha=args.lora_alpha,
                lora_target_modules=args.lora_target_modules,
                init_lora_weights=args.init_lora_weights,
                pretrained_lora_path=args.pretrained_lora_path,
            )
        else:
            self.pipe.denoising_model().requires_grad_(True)

        # 2) 构造优化器，仅优化 LoRA 参数或全部参数
        trainable_params = filter(
            lambda p: p.requires_grad, self.pipe.denoising_model().parameters()
        )
        self.optimizer = torch.optim.AdamW(trainable_params, lr=args.learning_rate)

    def freeze_parameters(self):
        self.pipe.requires_grad_(False)
        self.pipe.eval()
        self.pipe.denoising_model().train()

    def add_lora_to_model(
        self,
        model,
        lora_rank=4,
        lora_alpha=4,
        lora_target_modules="q,k,v,o,ffn.0,ffn.2",
        init_lora_weights="kaiming",
        pretrained_lora_path=None,
    ):
        if init_lora_weights == "kaiming":
            init_lora_weights = True
        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            init_lora_weights=init_lora_weights,
            target_modules=lora_target_modules.split(","),
        )
        model = inject_adapter_in_model(lora_config, model)
        for param in model.parameters():
            if param.requires_grad:
                param.data = param.data.to(torch.float32)
        if pretrained_lora_path is not None:
            state_dict = load_state_dict(pretrained_lora_path)
            missing_keys, unexpected_keys = model.load_state_dict(
                state_dict, strict=False
            )
            updated = len(list(model.named_parameters())) - len(missing_keys)
            print(
                f"{updated} 个参数从 LoRA 权重加载完成，{len(unexpected_keys)} 个键不匹配。"
            )

    def training_step(self, batch):
        latents = batch["latents"].to(self.accelerator.device)
        prompt_emb = batch["prompt_emb"]
        prompt_emb["context"] = prompt_emb["context"][0].to(self.accelerator.device)
        image_emb = batch.get("image_emb", {})
        if "clip_feature" in image_emb:
            image_emb["clip_feature"] = image_emb["clip_feature"][0].to(
                self.accelerator.device
            )
        if "y" in image_emb:
            image_emb["y"] = image_emb["y"][0].to(self.accelerator.device)

        self.pipe.device = self.accelerator.device
        noise = torch.randn_like(latents)
        timestep_id = torch.randint(0, self.pipe.scheduler.num_train_timesteps, (1,))
        timestep = self.pipe.scheduler.timesteps[timestep_id].to(
            dtype=self.pipe.torch_dtype, device=self.accelerator.device
        )
        extra_input = self.pipe.prepare_extra_input(latents)
        noisy_latents = self.pipe.scheduler.add_noise(latents, noise, timestep)
        target = self.pipe.scheduler.training_target(latents, noise, timestep)
        noise_pred = self.pipe.denoising_model()(
            noisy_latents,
            timestep=timestep,
            **prompt_emb,
            **extra_input,
            **image_emb,
            use_gradient_checkpointing=self.args.use_gradient_checkpointing,
            use_gradient_checkpointing_offload=self.args.use_gradient_checkpointing_offload,
        )
        loss = torch.nn.functional.mse_loss(noise_pred.float(), target.float())
        loss = loss * self.pipe.scheduler.training_weight(timestep)
        return loss

    def save_model(self, output_dir):
        # ZeRO-3 下，使用 accelerator.save 来保存当前 rank 中的所有参数
        os.makedirs(output_dir, exist_ok=True)
        # 保存模型的 state_dict 到 pytorch_model.bin
        save_path = os.path.join(output_dir, "pytorch_model.bin")
        self.accelerator.save(self.pipe.denoising_model().state_dict(), save_path)


def parse_args():
    parser = argparse.ArgumentParser(
        description="WanVideo Accelerate + DeepSpeed 训练示例"
    )
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, default="./log")
    parser.add_argument("--dit_path", type=str, required=True)
    parser.add_argument("--vae_path", type=str, required=True)
    parser.add_argument("--image_encoder_path", type=str, default=None)
    parser.add_argument("--steps_per_epoch", type=int, default=100)
    parser.add_argument("--max_epochs", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--accumulate_grad_batches", type=int, default=1)
    parser.add_argument("--lora_rank", type=int, default=4)
    parser.add_argument("--lora_alpha", type=float, default=4.0)
    parser.add_argument(
        "--train_architecture", type=str, default="lora", choices=["lora", "full"]
    )
    parser.add_argument(
        "--lora_target_modules", type=str, default="q,k,v,o,ffn.0,ffn.2"
    )
    parser.add_argument(
        "--init_lora_weights",
        type=str,
        default="kaiming",
        choices=["gaussian", "kaiming"],
    )
    parser.add_argument("--use_gradient_checkpointing", action="store_true")
    parser.add_argument("--use_gradient_checkpointing_offload", action="store_true")
    parser.add_argument("--dataloader_num_workers", type=int, default=4)
    return parser.parse_args()


def main():
    args = parse_args()
    accelerator = Accelerator()
    device = accelerator.device

    # 1) 构造 Dataset 和 DataLoader
    dataset = TensorDataset(
        args.dataset_path,
        os.path.join(args.dataset_path, "metadata.csv"),
        steps_per_epoch=args.steps_per_epoch,
        num_frames=81,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        shuffle=True,
        batch_size=1,
        num_workers=args.dataloader_num_workers,
    )

    # 2) 初始化 WanTrainer 并将模型、优化器、dataloader 等注册到 Accelerate
    trainer = WanTrainer(args)
    model = trainer
    optimizer = trainer.optimizer

    model.pipe.denoising_model().to(device)
    model.accelerator = accelerator
    # dit = model.pipe.dit
    # 使用 accelerate 包装 model 和 optimizer 和 dataloader
    model.pipe.dit, optimizer, dataloader = accelerator.prepare(
        model.pipe.dit, optimizer, dataloader
    )
    # model.pipe.denoising_model() = dit_t
    # model.pipe.dit= dit_t

    # 3) 训练循环
    global_step = 0
    for epoch in range(args.max_epochs):
        for step, batch in enumerate(dataloader):
            with accelerator.accumulate(model.pipe.denoising_model()):
                loss = trainer.training_step(batch)
                accelerator.backward(loss)
                accelerator.clip_grad_norm_(
                    model.pipe.denoising_model().parameters(), 1.0
                )
                optimizer.step()
                optimizer.zero_grad()
            global_step += 1
            if global_step % 100 == 0:
                print(f"Epoch {epoch}, Step {global_step}, Loss: {loss.item():.4f}")

    # 4) 保存最终权重
    trainer.save_model(args.output_path)


if __name__ == "__main__":
    main()
