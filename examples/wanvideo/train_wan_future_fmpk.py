import torch, os, imageio, argparse
from torchvision.transforms import v2
from einops import rearrange
import lightning as pl
import pandas as pd
from diffsynth import WanVideoPipeline, ModelManager, load_state_dict
from peft import LoraConfig, inject_adapter_in_model
import torchvision
from PIL import Image
import numpy as np
import random
import datetime
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import random_split
from skvideo.io import vwrite
from tqdm import tqdm
import time
from pytorch_lightning.utilities import rank_zero_only
import torch.distributed as dist
from tabulate import tabulate
import wandb
import yaml
from dataset import TensorHierachicalDataset
import shutil
import utils


def set_seed(seed: int = 42):
    # 1. Python、NumPy、Torch 全局随机种子
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # 2. CUDA 后端设置：禁用非确定性算法，禁止 benchmark
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # 3. （可选）确保新的 cuBLAS workspace 行为更可控
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    # 4. Lightning 自带的全局种子设置
    pl.seed_everything(seed, workers=True)


def save_param_table(model, base_lr, clean_lr, out_path="params_table.txt"):
    """
    将模型中所有参数的 name、learning rate、requires_grad 写成表格并保存为 txt。

    :param model: 包含 named_parameters() 的模型（如 self.pipe.denoising_model()）
    :param base_lr: 其他参数的基础学习率（如 self.learning_rate）
    :param clean_lr: clean_x_embedder 模块的学习率（如 1e-4）
    :param out_path: 保存表格的文件路径
    """
    # 收集 rows
    rows = []
    for name, param in model.named_parameters():
        lr = clean_lr if "clean_x_embedder" in name else base_lr
        rows.append([name, lr, param.requires_grad])

    # 生成 GitHub 风格的表格
    table = tabulate(
        rows,
        headers=["parameter name", "learning rate", "requires_grad"],
        tablefmt="github",
    )

    # 确保目录存在
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    # 写入文件
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(table)
    print(f"Saved parameter table to {out_path}")


# TODO move above helper functions to utils


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
        if (
            reader.count_frames() < max_num_frames
            or reader.count_frames() - 1 < start_frame_id + (num_frames - 1) * interval
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
        file_ext_name = file_path.split(".")[-1]
        if file_ext_name.lower() in ["jpg", "jpeg", "png", "webp"]:
            return True
        return False

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
                raise ValueError(
                    f"{path} is not a video. I2V model doesn't support image-to-image training."
                )
            video = self.load_image(path)
        else:
            video = self.load_video(path)
        if self.is_i2v:
            video, first_frame = video
            data = {
                "text": text,
                "video": video,
                "path": path,
                "first_frame": first_frame,
            }
        else:
            data = {"text": text, "video": video, "path": path}
        return data

    def __len__(self):
        return len(self.path)


class LightningModelForDataProcess(pl.LightningModule):
    def __init__(
        self,
        text_encoder_path,
        vae_path,
        image_encoder_path=None,
        tiled=False,
        tile_size=(34, 34),
        tile_stride=(18, 16),
    ):
        super().__init__()
        model_path = [text_encoder_path, vae_path]
        if image_encoder_path is not None:
            model_path.append(image_encoder_path)
        model_manager = ModelManager(torch_dtype=torch.bfloat16, device="cpu")
        model_manager.load_models(model_path)
        self.pipe = WanVideoPipeline.from_model_manager(model_manager)

        self.tiler_kwargs = {
            "tiled": tiled,
            "tile_size": tile_size,
            "tile_stride": tile_stride,
        }

    def test_step(self, batch, batch_idx):
        text, video, path = batch["text"][0], batch["video"], batch["path"][0]

        self.pipe.device = self.device
        if video is not None:
            # prompt
            prompt_emb = self.pipe.encode_prompt(text)
            # video
            video = video.to(dtype=self.pipe.torch_dtype, device=self.pipe.device)
            latents = self.pipe.encode_video(video, **self.tiler_kwargs)[0]
            # start_latent = self.pipe.encode_video(video[:, :, :1], **self.tiler_kwargs)[
            #     0
            # ]
            # image
            if "first_frame" in batch:
                first_frame = Image.fromarray(batch["first_frame"][0].cpu().numpy())
                _, _, num_frames, height, width = video.shape
                image_emb = self.pipe.encode_image(
                    first_frame, None, num_frames, height, width
                )
            else:
                image_emb = {}
            data = {
                "latents": latents,
                # "start_latent": start_latent,
                "prompt_emb": prompt_emb,
                "image_emb": image_emb,
            }
            torch.save(data, path + ".tensors.pth")


class LightningModelForTrain(pl.LightningModule):
    def __init__(
        self,
        dit_path,
        vae_path,
        run_dir,
        trained_dit_path=None,
        learning_rate=1e-5,
        lora_rank=4,
        lora_alpha=4,
        num_validation_blocks=5,
        train_architecture="lora",
        lora_target_modules="q,k,v,o,ffn.0,ffn.2",
        init_lora_weights="kaiming",
        use_gradient_checkpointing=True,
        use_gradient_checkpointing_offload=False,
        pretrained_lora_path=None,
        args=None,
    ):
        super().__init__()
        model_manager = ModelManager(torch_dtype=torch.bfloat16, device="cpu")
        if dit_path != None:
            if os.path.exists(dit_path):
                model_manager.load_models([vae_path, dit_path])
            elif "," in dit_path:
                dit_path = dit_path.split(",")
                model_manager.load_models([vae_path, dit_path])
        elif dit_path == None:
            if trained_dit_path != None:
                model_manager.load_models([vae_path, trained_dit_path])
            else:
                raise Exception("at least provide dit path or trained dit path")

        self.run_dir = run_dir
        self.pipe = WanVideoPipeline.from_model_manager(model_manager)
        self.pipe.scheduler.set_timesteps(1000, training=True)
        self.freeze_parameters()
        if train_architecture == "lora":
            self.add_lora_to_model(
                self.pipe.denoising_model(),
                lora_rank=lora_rank,
                lora_alpha=lora_alpha,
                lora_target_modules=lora_target_modules,
                init_lora_weights=init_lora_weights,
                pretrained_lora_path=pretrained_lora_path,
            )
        else:
            self.pipe.denoising_model().requires_grad_(True)

        self.learning_rate = learning_rate
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload
        self.num_validation_blocks = num_validation_blocks
        self.args = args

    def freeze_parameters(self):
        # Freeze parameters
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
        state_dict_converter=None,
    ):
        # Add LoRA to UNet
        self.lora_alpha = lora_alpha
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
            # Upcast LoRA parameters into fp32
            if param.requires_grad:
                param.data = param.to(torch.float32)

        # Lora pretrained lora weights
        if pretrained_lora_path is not None:
            state_dict = load_state_dict(pretrained_lora_path)
            if state_dict_converter is not None:
                state_dict = state_dict_converter(state_dict)
            missing_keys, unexpected_keys = model.load_state_dict(
                state_dict, strict=False
            )
            all_keys = [i for i, _ in model.named_parameters()]
            num_updated_keys = len(all_keys) - len(missing_keys)
            num_unexpected_keys = len(unexpected_keys)
            print(
                f"{num_updated_keys} parameters are loaded from {pretrained_lora_path}. {num_unexpected_keys} parameters are unexpected."
            )

    @rank_zero_only
    def preview_dataset(self, latents, latents_indice_info, path):
        m_latents = latents[:, :, :6]
        s_latents = latents[:, :, 6:]
        tiler_kwargs = {"tiled": True, "tile_size": (30, 52), "tile_stride": (15, 26)}
        frames_m = self.pipe.decode_video(
            m_latents.to("cpu").to(torch.bfloat16), **tiler_kwargs
        )
        frames_m = self.pipe.tensor2video(frames_m[0])
        vwrite(
            f"{self.run_dir}/check_m.mp4",
            frames_m,
        )
        s_latents_video = None

        for l in range(3):
            lat_s = s_latents[:, :, l].unsqueeze(2)
            lat_s = rearrange(
                lat_s,
                " b o 1 (h p1) (w p2) -> b o (p1 p2) h w",
                p1=2,
                p2=2,
            )
            s_latents_video = (
                lat_s
                if s_latents_video == None
                else torch.cat((s_latents_video, lat_s), dim=2)
            )
        frames_s = self.pipe.decode_video(
            s_latents_video.to("cpu").to(torch.bfloat16), **tiler_kwargs
        )
        frames_s = self.pipe.tensor2video(frames_s[0])
        vwrite(
            f"{self.run_dir}/check_s.mp4",
            frames_s,
        )
        utils.extract_frames(path, 4, self.run_dir)
        # pass

    def training_step(self, batch, batch_idx):
        # Data
        # print(batch_idx)
        if batch_idx == 0:
            self.preview_dataset(
                batch["latents"], batch["latents_indice_info"], batch["path"][0]
            )
        latents = batch["latents"].to(self.device)
        prompt_emb = batch["prompt_emb"]
        prompt_emb["context"] = prompt_emb["context"][0].to(self.device)
        image_emb = batch["image_emb"]
        if "clip_feature" in image_emb:
            image_emb["clip_feature"] = image_emb["clip_feature"][0].to(self.device)
        if "y" in image_emb:
            image_emb["y"] = image_emb["y"][0].to(self.device)

        # Loss
        self.pipe.device = self.device
        noise = torch.randn_like(latents)
        timestep_id = torch.randint(0, self.pipe.scheduler.num_train_timesteps, (1,))
        timestep = self.pipe.scheduler.timesteps[timestep_id].to(
            dtype=self.pipe.torch_dtype, device=self.pipe.device
        )
        extra_input = self.pipe.prepare_extra_input(latents)
        noisy_latents = self.pipe.scheduler.add_noise(latents, noise, timestep)
        training_target = self.pipe.scheduler.training_target(latents, noise, timestep)

        clean_latent_indices_start = batch["clean_latent_indices_start"].to(self.device)
        clean_latent_4x_indices = batch["clean_latent_4x_indices"].to(self.device)
        clean_latent_2x_indices = batch["clean_latent_2x_indices"].to(self.device)
        clean_latent_indices = batch["clean_latent_indices"].to(self.device)
        latents_indice_info = batch["latents_indice_info"]
        start_latent = batch["start_latent"].to(self.device)
        clean_latents_4x = batch["clean_latents_4x"].to(self.device)
        clean_latents_2x = batch["clean_latents_2x"].to(self.device)
        clean_latents = batch["clean_latents"].to(self.device)
        # latents = batch["latents"]
        # Compute loss
        noise_pred = self.pipe.denoising_model()(
            noisy_latents,
            timestep=timestep,
            **prompt_emb,
            **extra_input,
            **image_emb,
            use_gradient_checkpointing=self.use_gradient_checkpointing,
            use_gradient_checkpointing_offload=self.use_gradient_checkpointing_offload,
            clean_latent_indices_start=clean_latent_indices_start,
            clean_latent_4x_indices=clean_latent_4x_indices,
            clean_latent_2x_indices=clean_latent_2x_indices,
            clean_latent_indices=clean_latent_indices,
            latents_indice_info=latents_indice_info,
            start_latent=start_latent,
            clean_latents_4x=clean_latents_4x,
            clean_latents_2x=clean_latents_2x,
            clean_latents=clean_latents,
        )
        loss = torch.nn.functional.mse_loss(noise_pred.float(), training_target.float())
        loss = loss * self.pipe.scheduler.training_weight(timestep)

        # Record log
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        named_trainable = [
            (name, param)
            for name, param in self.pipe.denoising_model().named_parameters()
            if param.requires_grad
        ]
        clean_params = [p for n, p in named_trainable if "clean_x_embedder" in n]
        other_params = [p for n, p in named_trainable if "clean_x_embedder" not in n]
        param_groups = [
            {"params": clean_params, "lr": 1e-4},
            {"params": other_params, "lr": self.learning_rate},
        ]
        optimizer = torch.optim.AdamW(param_groups)
        save_param_table(
            model=self.pipe.denoising_model(),
            base_lr=self.learning_rate,
            clean_lr=1e-4,
            out_path=f"{self.run_dir}/param_table.txt",
        )
        return optimizer

    def on_save_checkpoint(self, checkpoint):
        checkpoint.clear()
        trainable_param_names = list(
            filter(
                lambda named_param: named_param[1].requires_grad,
                self.pipe.denoising_model().named_parameters(),
            )
        )
        trainable_param_names = set(
            [named_param[0] for named_param in trainable_param_names]
        )
        state_dict = self.pipe.denoising_model().state_dict()
        lora_state_dict = {}
        for name, param in state_dict.items():
            if name in trainable_param_names:
                lora_state_dict[name] = param
        checkpoint.update(lora_state_dict)

    def validation_step(self, batch, batch_idx):
        predicting_indice = np.array(
            [int(ids) for ids in self.args.predicting_indice.split(",")]
        )
        self.pipe.requires_grad_(False)
        self.pipe.eval()
        start_latent = batch["start_latent"].to(self.device)
        history_latents = torch.zeros(
            size=(1, 16, 16 + 2 + 1, start_latent.size(-2), start_latent.size(-1)),
            dtype=torch.float32,
        ).cpu()
        history_pixels = None
        s_latents = torch.zeros(
            size=(
                1,
                16,
                16 + 2 + 1,
                start_latent.size(-2) // 2,
                start_latent.size(-1) // 2,
            ),
            dtype=torch.float32,
        ).cpu()
        history_latents = torch.cat(
            [history_latents, start_latent.to(history_latents)], dim=2
        )
        total_generated_latent_frames = 1
        clean_latent_indices_start = batch["clean_latent_indices_start"].to(self.device)
        clean_latent_4x_indices = batch["clean_latent_4x_indices"].to(self.device)
        clean_latent_2x_indices = batch["clean_latent_2x_indices"].to(self.device)
        clean_latent_indices = batch["clean_latent_indices"].to(self.device)
        latents_indice_info = batch["latents_indice_info"]
        prompt_emb, image_emb = batch["prompt_emb"], batch["image_emb"]
        prompt_emb["context"] = prompt_emb["context"][0].to(self.device)
        image_emb = batch["image_emb"]
        if "clip_feature" in image_emb:
            image_emb["clip_feature"] = image_emb["clip_feature"][0].to(self.device)
        if "y" in image_emb:
            image_emb["y"] = image_emb["y"][0].to(self.device)

        for section_index in tqdm(range(self.num_validation_blocks)):
            clean_latents_4x, clean_latents_2x, clean_latents_1x = history_latents[
                :, :, -sum([16, 2, 1]) :, :, :
            ].split([16, 2, 1], dim=2)
            clean_latents = torch.cat(
                [start_latent.to(history_latents), clean_latents_1x], dim=2
            )
            # latents
            indice_drift = (
                section_index * len(predicting_indice)
                if self.args.is_absolute_rope
                else 0
            )
            clean_latent_indices[:, :, 1] += indice_drift
            clean_latent_kwargs = {
                "clean_latent_indices_start": clean_latent_indices_start,
                "clean_latent_4x_indices": clean_latent_4x_indices + indice_drift,
                "clean_latent_2x_indices": clean_latent_2x_indices + indice_drift,
                "clean_latent_indices": clean_latent_indices,
                "latents_indice_info": latents_indice_info,
                "start_latent": start_latent,
                "clean_latents_4x": clean_latents_4x,
                "clean_latents_2x": clean_latents_2x,
                "clean_latents": clean_latents,
                # "latents": latents,
            }
            noise = torch.randn(
                batch["latents"].shape[:],
                generator=torch.Generator().manual_seed(0),
            )
            video, generated_latents, tiler_kwargs = self.pipe(
                cfg_scale=1.0,
                num_inference_steps=50,
                seed=0,
                tiled=True,
                noise=noise,
                prompt_emb=prompt_emb,
                image_emb=image_emb,
                clean_latent_kwargs=clean_latent_kwargs,
                device=torch.device("cuda"),
            )
            # self.tiler_kwargs = tiler_kwargs
            total_generated_latent_frames += int(generated_latents.shape[2])
            history_latents = torch.cat(
                [
                    history_latents,
                    generated_latents[:, :, predicting_indice].to(history_latents),
                ],
                dim=2,
            )
            for l in range(3):
                lat_s = generated_latents[:, :, 3 - l].to(history_latents).unsqueeze(2)
                lat_s = rearrange(
                    lat_s,
                    " b o 1 (h p1) (w p2) -> b o (p1 p2) h w",
                    p1=2,
                    p2=2,
                )
                s_latents = torch.cat((s_latents, lat_s), 2)
                # videos.append(latent)
        # print(f"saving to {self.run_dir}/{self.trainer.current_epoch}.mp4")
        # for i, v in enumerate(videos):
        #     videos[i] = np.asarray(v)
        p = batch["path"][0].replace("/", "_")
        frames_s = self.pipe.decode_video(
            s_latents[:, :, 16 + 2 + 1 :].to(torch.bfloat16), **tiler_kwargs
        )
        frames_s = self.pipe.tensor2video(frames_s[0])
        vwrite(
            f"{self.run_dir}/last3_{self.trainer.current_epoch}_{p}_{time.time_ns()}.mp4",
            frames_s,
        )
        self.pipe.load_models_to_device(["vae"])
        frames = self.pipe.decode_video(
            history_latents[:, :, 16 + 2 + 1 :].to(torch.bfloat16), **tiler_kwargs
        )
        self.pipe.load_models_to_device([])
        frames = self.pipe.tensor2video(frames[0])
        vwrite(
            f"{self.run_dir}/decode_all_{self.trainer.current_epoch}_{p}_{time.time_ns()}.mp4",
            frames,
        )
        self.pipe.scheduler.set_timesteps(1000, training=True)
        self.freeze_parameters()
        if False:
            self.add_lora_to_model(
                self.pipe.denoising_model(),
                lora_rank=lora_rank,
                lora_alpha=lora_alpha,
                lora_target_modules=lora_target_modules,
                init_lora_weights=init_lora_weights,
                pretrained_lora_path=pretrained_lora_path,
            )
        else:
            self.pipe.denoising_model().requires_grad_(True)
        return


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--task",
        type=str,
        default="data_process",
        required=True,
        choices=["data_process", "train", "inference"],
        help="Task. `data_process` or `train`.",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=None,
        required=True,
        help="The path of the Dataset.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="./",
        help="Path to save the model.",
    )
    parser.add_argument(
        "--text_encoder_path",
        type=str,
        default=None,
        help="Path of text encoder.",
    )
    parser.add_argument(
        "--image_encoder_path",
        type=str,
        default=None,
        help="Path of image encoder.",
    )
    parser.add_argument(
        "--vae_path",
        type=str,
        default=None,
        help="Path of VAE.",
    )
    parser.add_argument(
        "--dit_path",
        type=str,
        default=None,
        help="Path of DiT.",
    )
    parser.add_argument(
        "--trained_dit_path",
        type=str,
        default=None,
        help="Path of trained DiT, to surpass the initialization of DiT.",
    )
    parser.add_argument(
        "--tiled",
        default=False,
        action="store_true",
        help="Whether enable tile encode in VAE. This option can reduce VRAM required.",
    )
    parser.add_argument(
        "--tile_size_height",
        type=int,
        default=34,
        help="Tile size (height) in VAE.",
    )
    parser.add_argument(
        "--tile_size_width",
        type=int,
        default=34,
        help="Tile size (width) in VAE.",
    )
    parser.add_argument(
        "--tile_stride_height",
        type=int,
        default=18,
        help="Tile stride (height) in VAE.",
    )
    parser.add_argument(
        "--tile_stride_width",
        type=int,
        default=16,
        help="Tile stride (width) in VAE.",
    )
    parser.add_argument(
        "--steps_per_epoch",
        type=int,
        default=500,
        help="Number of steps per epoch.",
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=81,
        help="Number of frames.(in real)",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=480,
        help="Image height.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=832,
        help="Image width.",
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=1,
        help="Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-5,
        help="Learning rate.",
    )
    parser.add_argument(
        "--accumulate_grad_batches",
        type=int,
        default=1,
        help="The number of batches in gradient accumulation.",
    )
    parser.add_argument(
        "--max_epochs",
        type=int,
        default=1,
        help="Number of epochs.",
    )
    parser.add_argument(
        "--lora_target_modules",
        type=str,
        default="q,k,v,o,ffn.0,ffn.2",
        help="Layers with LoRA modules.",
    )
    parser.add_argument(
        "--init_lora_weights",
        type=str,
        default="kaiming",
        choices=["gaussian", "kaiming"],
        help="The initializing method of LoRA weight.",
    )
    parser.add_argument(
        "--training_strategy",
        type=str,
        default="auto",
        choices=["auto", "deepspeed_stage_1", "deepspeed_stage_2", "deepspeed_stage_3"],
        help="Training strategy",
    )
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=4,
        help="The dimension of the LoRA update matrices.",
    )
    parser.add_argument(
        "--lora_alpha",
        type=float,
        default=4.0,
        help="The weight of the LoRA update matrices.",
    )
    parser.add_argument(
        "--num_validation_blocks",
        type=int,
        default=5,
        help="num_validation_blocks",
    )
    parser.add_argument(
        "--use_gradient_checkpointing",
        default=False,
        action="store_true",
        help="Whether to use gradient checkpointing.",
    )
    parser.add_argument(
        "--use_gradient_checkpointing_offload",
        default=False,
        action="store_true",
        help="Whether to use gradient checkpointing offload.",
    )
    parser.add_argument(
        "--train_architecture",
        type=str,
        default="lora",
        choices=["lora", "full"],
        help="Model structure to train. LoRA training or full training.",
    )
    parser.add_argument(
        "--pretrained_lora_path",
        type=str,
        default=None,
        help="Pretrained LoRA path. Required if the training is resumed.",
    )
    parser.add_argument(
        "--s_path",
        type=str,
        default=None,
        help="Pretrained LoRA path. Required if the training is resumed.",
    )
    parser.add_argument(
        "--xs_path",
        type=str,
        default=None,
        help="Pretrained LoRA path. Required if the training is resumed.",
    )
    parser.add_argument(
        "--predict_config",
        type=str,
        default=None,
        help="Pretrained LoRA path. Required if the training is resumed.",
    )
    parser.add_argument(
        "--predicting_indice",
        type=str,
        default=None,
        help="Pretrained LoRA path. Required if the training is resumed.",
    )
    parser.add_argument(
        "--use_swanlab",
        default=False,
        action="store_true",
        help="Whether to use SwanLab logger.",
    )
    parser.add_argument(
        "--swanlab_mode",
        default=None,
        help="SwanLab mode (cloud or local).",
    )
    parser.add_argument(
        "--is_absolute_rope",
        default=False,
        action="store_true",
        help="Whether to use absolute rope.",
    )
    args = parser.parse_args()
    return args


def data_process(args):
    dataset = TextVideoDataset(
        args.dataset_path,
        os.path.join(args.dataset_path, "metadata.csv"),
        max_num_frames=args.num_frames,
        frame_interval=1,
        num_frames=args.num_frames,
        height=args.height,
        width=args.width,
        is_i2v=args.image_encoder_path is not None,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset, shuffle=False, batch_size=1, num_workers=args.dataloader_num_workers
    )
    model = LightningModelForDataProcess(
        text_encoder_path=args.text_encoder_path,
        image_encoder_path=args.image_encoder_path,
        vae_path=args.vae_path,
        tiled=args.tiled,
        tile_size=(args.tile_size_height, args.tile_size_width),
        tile_stride=(args.tile_stride_height, args.tile_stride_width),
    )
    trainer = pl.Trainer(
        accelerator="gpu",
        devices="auto",
        default_root_dir=args.output_path,
    )
    trainer.test(model, dataloader)


def save_args_to_yaml(args: argparse.Namespace, filepath: str):
    """
    将 argparse.Namespace 对象保存为 YAML 文件。

    :param args: argparse.parse_args() 返回的 Namespace 对象
    :param filepath: 要保存的 YAML 文件路径（可以是相对或绝对路径）
    """
    # 确保目录存在
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    # 把 Namespace 转为字典，然后写入 YAML
    with open(filepath, "w", encoding="utf-8") as f:
        yaml.safe_dump(vars(args), f, allow_unicode=True)


# @rank_zero_only
def init_wandb_logger(args):
    # 只会在 global_rank==0 时运行

    run_name = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M")
    run_name = f"{run_name}_lr_{args.learning_rate}_bsz_{args.accumulate_grad_batches * torch.cuda.device_count()}_eplen_{args.steps_per_epoch}"
    run_dir = os.path.join(args.output_path, run_name)
    save_args_to_yaml(args, os.path.join(run_dir, "args.yaml"))
    os.makedirs(run_dir, exist_ok=True)
    return (
        WandbLogger(
            project="Framepack_reproduce",
            name=run_name,
            id=run_name,  # 同 name 保证唯一
            save_dir="logs",
            entity="qianrj_team",
            config=vars(args),
        ),
        run_dir,
        run_name,
    )


@rank_zero_only
def update_config(logger, config_dict):
    logger.experiment.config.update(config_dict, allow_val_change=True)


@rank_zero_only
def get_folder_size(folder_path):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(folder_path):
        for filename in filenames:
            file_path = os.path.join(dirpath, filename)
            total_size += os.path.getsize(file_path)
    return total_size


@rank_zero_only
def delete_folders_under_size(root_folder, min_size=5):
    min_size = min_size * 1024 * 1024
    for dirpath, dirnames, filenames in os.walk(root_folder):
        for dirname in dirnames:
            if dirname == "tensorboard_logs":
                continue
            folder_path = os.path.join(dirpath, dirname)
            folder_size = get_folder_size(folder_path)
            if folder_size < min_size:
                print(f"Deleting folder: {folder_path}")
                try:
                    shutil.rmtree(folder_path)
                    print(f"Folder {folder_path} deleted successfully!")
                except OSError as e:
                    print(f"Error deleting folder {folder_path}: {e}")


def train(args):
    delete_folders_under_size(args.output_path)
    set_seed(3407)
    dataset = TensorHierachicalDataset(
        base_path=args.dataset_path,
        metadata_path=os.path.join(args.dataset_path, "metadata.csv"),
        steps_per_epoch=args.steps_per_epoch,
        episode_length_real=args.num_frames,
        s_path=args.s_path,
        xs_path=args.xs_path,
        predict_config=args.predict_config,
        is_absolute_rope=args.is_absolute_rope,
    )
    val_dataset = TensorHierachicalDataset(
        base_path=args.dataset_path,
        metadata_path=os.path.join(args.dataset_path, "metadata.csv"),
        steps_per_epoch=args.steps_per_epoch,
        episode_length_real=args.num_frames,
        s_path=args.s_path,
        xs_path=args.xs_path,
        predict_config=args.predict_config,
        is_val_dataset=True,
        is_absolute_rope=args.is_absolute_rope,
    )
    # trainset, valset = random_split(dataset, [0.9, 0.1])

    dataloader = torch.utils.data.DataLoader(
        dataset, shuffle=True, batch_size=1, num_workers=args.dataloader_num_workers
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        shuffle=False,
        batch_size=1,
        num_workers=args.dataloader_num_workers,
    )

    setattr(args, "predict_config", dataset.predict_config)
    wandb_logger, run_dir, run_name = init_wandb_logger(args)
    # name_holder = [wandb_logger, run_name, run_dir]
    # dist.broadcast_object_list(name_holder, src=0)
    # wandb_logger, run_name, run_dir = name_holder[0], name_holder[1], name_holder[2]
    update_config(wandb_logger, vars(args))

    model = LightningModelForTrain(
        dit_path=args.dit_path,
        vae_path=args.vae_path,
        run_dir=run_dir,
        trained_dit_path=args.trained_dit_path,
        learning_rate=args.learning_rate,
        train_architecture=args.train_architecture,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_target_modules=args.lora_target_modules,
        init_lora_weights=args.init_lora_weights,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload,
        pretrained_lora_path=args.pretrained_lora_path,
        num_validation_blocks=args.num_validation_blocks,
        args=args,
    )
    # if args.use_swanlab:
    #     from swanlab.integration.pytorch_lightning import SwanLabLogger

    #     swanlab_config = {"UPPERFRAMEWORK": "DiffSynth-Studio"}
    #     swanlab_config.update(vars(args))
    #     swanlab_logger = SwanLabLogger(
    #         project="wan",
    #         name="wan",
    #         config=swanlab_config,
    #         mode=args.swanlab_mode,
    #         logdir=os.path.join(args.output_path, "swanlog"),
    #     )
    #     logger = [swanlab_logger]
    # else:
    #     logger = None

    # wandb.config.update(vars(args))
    # logger = wandb_logger
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator="gpu",
        devices="auto",
        precision="bf16",
        strategy=args.training_strategy,
        default_root_dir=run_dir,
        accumulate_grad_batches=args.accumulate_grad_batches,
        callbacks=[
            pl.pytorch.callbacks.ModelCheckpoint(save_last=True, every_n_epochs=5)
        ],
        logger=wandb_logger,
        log_every_n_steps=1,
        # gradient_clip_val=0.5,  # clip 阈值
        # gradient_clip_algorithm="norm",
        num_sanity_val_steps=1,
        limit_val_batches=1,
    )
    trainer.fit(model, dataloader, val_dataloaders=val_dataloader)


def inference(args):
    set_seed(3407)
    dataset = TensorHierachicalDataset(
        base_path=args.dataset_path,
        metadata_path=os.path.join(args.dataset_path, "metadata.csv"),
        steps_per_epoch=args.steps_per_epoch,
        episode_length_real=args.num_frames,
        s_path=args.s_path,
        xs_path=args.xs_path,
        predict_config=args.predict_config,
    )
    val_dataset = TensorHierachicalDataset(
        base_path=args.dataset_path,
        metadata_path=os.path.join(args.dataset_path, "metadata.csv"),
        steps_per_epoch=args.steps_per_epoch,
        episode_length_real=args.num_frames,
        s_path=args.s_path,
        xs_path=args.xs_path,
        predict_config=args.predict_config,
        is_val_dataset=True,
    )
    # trainset, valset = random_split(dataset, [0.9, 0.1])

    dataloader = torch.utils.data.DataLoader(
        dataset, shuffle=True, batch_size=1, num_workers=args.dataloader_num_workers
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        shuffle=False,
        batch_size=1,
        num_workers=args.dataloader_num_workers,
    )
    wandb_logger, run_dir, run_name = init_wandb_logger(args)
    # name_holder = [wandb_logger, run_name, run_dir]
    # dist.broadcast_object_list(name_holder, src=0)
    # wandb_logger, run_name, run_dir = name_holder[0], name_holder[1], name_holder[2]
    update_config(wandb_logger, vars(args))

    model = LightningModelForTrain(
        trained_dit_path=args.trained_dit_path,
        dit_path=args.dit_path,
        vae_path=args.vae_path,
        run_dir=run_dir,
        learning_rate=args.learning_rate,
        train_architecture=args.train_architecture,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_target_modules=args.lora_target_modules,
        init_lora_weights=args.init_lora_weights,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload,
        pretrained_lora_path=args.pretrained_lora_path,
    )
    if args.use_swanlab:
        from swanlab.integration.pytorch_lightning import SwanLabLogger

        swanlab_config = {"UPPERFRAMEWORK": "DiffSynth-Studio"}
        swanlab_config.update(vars(args))
        swanlab_logger = SwanLabLogger(
            project="wan",
            name="wan",
            config=swanlab_config,
            mode=args.swanlab_mode,
            logdir=os.path.join(args.output_path, "swanlog"),
        )
        logger = [swanlab_logger]
    else:
        logger = None

    # wandb.config.update(vars(args))
    # logger = wandb_logger
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator="gpu",
        devices="auto",
        precision="bf16",
        strategy=args.training_strategy,
        default_root_dir=run_dir,
        accumulate_grad_batches=args.accumulate_grad_batches,
        callbacks=[pl.pytorch.callbacks.ModelCheckpoint(save_top_k=-1)],
        logger=wandb_logger,
        log_every_n_steps=1,
        # gradient_clip_val=0.5,  # clip 阈值
        # gradient_clip_algorithm="norm",
        limit_train_batches=0,
        num_sanity_val_steps=1,
        limit_val_batches=1,
    )
    trainer.fit(model, dataloader, val_dataloaders=val_dataloader)


if __name__ == "__main__":
    args = parse_args()
    if args.task == "data_process":
        data_process(args)
    elif args.task == "train":
        train(args)
    elif args.task == "inference":
        inference(args)
