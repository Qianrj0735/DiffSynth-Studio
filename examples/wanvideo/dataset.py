import torch
import pandas as pd
import os
import random
import numpy as np
import yaml
from utils import *
from einops import rearrange


class TensorHierachicalDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        base_path,
        s_path,
        xs_path,
        ms_path,
        predict_config,
        metadata_path,
        steps_per_epoch,
        episode_length_real=385,
        latent_window_size=9,
        is_val_dataset=False,
        is_absolute_rope=False,
    ):
        metadata = pd.read_csv(metadata_path)
        self.path = [
            os.path.join(base_path, "train", file_name)
            for file_name in metadata["file_name"]
        ]
        print(len(self.path), "videos in metadata.")
        self.path = [
            i + ".tensors.pth" for i in self.path if os.path.exists(i + ".tensors.pth")
        ]
        print(len(self.path), "tensors cached in metadata.")
        assert len(self.path) > 0

        self.steps_per_epoch = steps_per_epoch
        self.episode_length = (episode_length_real - 1) // 4
        self.latent_window_size = latent_window_size
        self.is_val_dataset = is_val_dataset
        self.s_path = s_path
        self.xs_path = xs_path
        self.ms_path = ms_path
        self.predict_config = load_config(predict_config)
        self.is_absolute_rope = is_absolute_rope

    def __getitem__(self, index):
        data_id = (
            torch.randint(0, len(self.path), (1,))[0] if not self.is_val_dataset else 0
        )
        data_id = (data_id + index) % len(self.path)  # For fixed seed.
        path = self.path[data_id]
        data_ms = torch.load(
            os.path.join(self.ms_path, "train", os.path.basename(path)),
            weights_only=True,
            map_location="cpu",
        )
        data_s = torch.load(
            os.path.join(self.s_path, "train", os.path.basename(path)),
            weights_only=True,
            map_location="cpu",
        )
        data_xs = torch.load(
            os.path.join(self.xs_path, "train", os.path.basename(path)),
            weights_only=True,
            map_location="cpu",
        )
        data = torch.load(path, weights_only=True, map_location="cpu")
        if self.is_val_dataset:
            context_data_id = data_id  # torch.randint(0, len(self.path), (1,))[0]
            context_data_id = (context_data_id + index) % len(
                self.path
            )  # For fixed seed.
            context_path = self.path[context_data_id]
            context_data = torch.load(
                context_path, weights_only=True, map_location="cpu"
            )
            data["prompt_emb"] = context_data["prompt_emb"]
            # data['image_emb']
        generating_first_idx = random.randint(
            1, self.episode_length - 1 - self.latent_window_size
        )
        if self.is_val_dataset:
            generating_first_idx = 1
        generating_indices = torch.arange(
            generating_first_idx, generating_first_idx + self.latent_window_size
        )
        indices = torch.arange(
            0, sum([1, 16, 2, 1, self.latent_window_size])
        ).unsqueeze(0)
        (
            clean_latent_indices_start,
            clean_latent_4x_indices,
            clean_latent_2x_indices,
            clean_latent_1x_indices,
            latent_indices,
        ) = indices.split([1, 16, 2, 1, self.latent_window_size], dim=1)
        read_clean_latent_4x_indices = (
            np.array(clean_latent_4x_indices[0]) - 20 + generating_first_idx
        )
        read_clean_latent_2x_indices = (
            np.array(clean_latent_2x_indices[0]) - 20 + generating_first_idx
        )
        read_clean_latent_1x_indices = (
            np.array(clean_latent_1x_indices[0]) - 20 + generating_first_idx
        )

        clean_latents_4x = self.read_latent(
            data["latents"], read_clean_latent_4x_indices
        )
        clean_latents_2x = self.read_latent(
            data["latents"], read_clean_latent_2x_indices
        )
        clean_latents_1x = self.read_latent(
            data["latents"], read_clean_latent_1x_indices
        )
        # latents = self.read_latent(data["latents"], generating_indices)
        latents, latents_indice_info = self.read_packed_latent(
            {
                "m": data["latents"],
                "ms": data_ms["latents"],
                "s": data_s["latents"],
                "xs": data_xs["latents"],
            },
            generating_indices,
        )

        if self.is_absolute_rope:
            indice_difference = int(latent_indices[0, 0] - generating_indices[0])
        else:
            indice_difference = 0

        clean_latent_4x_indices -= indice_difference
        clean_latent_2x_indices -= indice_difference
        clean_latent_1x_indices -= indice_difference
        latent_indices -= indice_difference

        start_latent = self.read_latent(data["latents"], np.zeros(1))
        clean_latents = torch.cat((start_latent, clean_latents_1x), dim=1)
        clean_latent_indices = torch.cat(
            [clean_latent_indices_start, clean_latent_1x_indices], dim=1
        )
        data["clean_latent_indices_start"] = clean_latent_indices_start
        data["clean_latent_4x_indices"] = clean_latent_4x_indices
        data["clean_latent_2x_indices"] = clean_latent_2x_indices
        data["clean_latent_indices"] = clean_latent_indices
        # data["latent_indices"] = latent_indices
        data["start_latent"] = start_latent
        data["clean_latents_4x"] = clean_latents_4x
        data["clean_latents_2x"] = clean_latents_2x
        data["clean_latents"] = clean_latents
        data["latents"] = latents
        data["latents_indice_info"] = latents_indice_info
        data["path"] = path if not self.is_val_dataset else context_path
        return data

    def read_packed_latent(self, latents_dict, read_indices):
        actual_indices_dict = compute_actual_indices(
            self.predict_config, read_indices[0]
        )
        read_blocks = {}
        for i, (block_name, read_indices) in enumerate(actual_indices_dict.items()):
            mode = self.predict_config[block_name]["mode"]
            l = self.read_latent(
                latents_dict[mode],
                np.array(read_indices),
            )
            if mode == "s":
                l = rearrange(l, "b (p1 p2) h w -> b 1 (h p1) (w p2)", p1=2, p2=2)
            elif mode == "xs":
                l = rearrange(l, "b (p1 p2) h w -> b 1 (h p1) (w p2)", p1=4, p2=4)
            read_blocks[i] = {"latents": l, "indices": read_indices, "mode": mode}
        return self.post_read_blocks(read_blocks)

    def post_read_blocks(self, read_blocks):
        string_info = ""
        latents = []
        for k, v in read_blocks.items():
            latents.append(v["latents"])
            for ind in v["indices"]:
                string_info += f"{v['mode']}_{ind},"
            string_info += "|"
        return torch.cat(latents, dim=1), string_info

    def read_latent(self, latents, read_indices):
        clean_latent = torch.zeros_like(latents[:, : len(read_indices)])
        clean_latent[:, read_indices >= 0] = latents[:, read_indices[read_indices >= 0]]
        return clean_latent

    def __len__(self):
        return self.steps_per_epoch


if __name__ == "__main__":
    # use the dataset defined above and enumerate the dataset:
    dataset = TensorHierachicalDataset(
        base_path="/home/workspace/DiffSynth-Studio/epic",
        s_path="/home/workspace/DiffSynth-Studio/epic_s",
        xs_path="/home/workspace/DiffSynth-Studio/epic_s",
        metadata_path="/home/workspace/DiffSynth-Studio/epic/metadata.csv",
        predict_config="examples/wanvideo/predict_pack_configs/3hrz.yml",
        steps_per_epoch=199,
        is_val_dataset=True,
    )  # 创建 TensorDataset 实例
    for idx, data in enumerate(dataset):  # 使用 enumerate 枚举数据集
        print(f"Index: {idx}, Data: {data}")  # 打印索引和数据
