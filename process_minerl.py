import os
import json
import csv
import cv2
from tqdm import tqdm
import imageio
from skvideo.io import vwrite
from multiprocessing import Pool, cpu_count


def video_to_frames_imageio(path: str) -> list:
    """
    用 imageio 读取视频，每帧自动返回为 RGB 格式的 ndarray。
    """
    reader = imageio.get_reader(path)
    frames = [frame for frame in reader]  # frame 是 (H, W, 3) ndarray
    reader.close()
    return frames


def process_clip(args):
    video_name, meta_dict, input_dir, save_dir, L = args
    input_path = os.path.join(input_dir, video_name)
    if not os.path.isfile(input_path):
        print(f"Warning: {video_name} 不存在，跳过。")
        return []

    frames = video_to_frames_imageio(input_path)
    rows = []
    base_name = os.path.splitext(video_name)[0]

    for f_idx in range(len(frames)):
        if f_idx + L > len(frames):
            break
        cliped_frames = frames[f_idx : f_idx + L]
        new_name = f"{base_name}_clip_{f_idx}_{L}.mp4"
        os.makedirs(os.path.join(save_dir, "train"), exist_ok=True)
        output_path = os.path.join(save_dir, "train", new_name)
        vwrite(output_path, cliped_frames)
        clipped_meta = {k: v[f_idx : f_idx + L] for k, v in meta_dict.items()}
        rows.append((new_name, str(clipped_meta)))
        break

    return rows


def clip_videos(
    input_dir: str, metadata_path: str, save_dir: str, L: int, num_workers: int = None
):
    """
    Parameters:
    - input_dir: 存放原始视频的文件夹
    - metadata_path: metadata.json 的路径
    - save_dir: 剪切后视频和 metadata.csv 的存放目录
    - L: 要剪切的帧数长度
    - num_workers: 并行进程数, 默认使用 cpu_count()
    """
    # 1. 读取 metadata.json
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    os.makedirs(save_dir, exist_ok=True)
    csv_path = os.path.join(save_dir, "metadata.csv")

    # 3. 准备多进程参数列表
    tasks = [
        (video_name, meta_dict, input_dir, save_dir, L)
        for video_name, meta_dict in metadata.items()
    ]

    # 4. 启动 Pool
    if num_workers is None:
        num_workers = cpu_count()
    with Pool(processes=num_workers) as pool, open(
        csv_path, "w", encoding="utf-8", newline=""
    ) as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["file_name", "text"])

        for rows in tqdm(pool.imap_unordered(process_clip, tasks), total=len(tasks)):
            for row in rows:
                writer.writerow(row)

    print(f"处理完成，所有文件已保存到：{save_dir}")


# 示例调用
if __name__ == "__main__":
    clip_videos(
        input_dir="minerl/minerl_navigate/train",
        metadata_path="minerl/minerl_navigate/train/metadata.json",
        save_dir="minerl_processed",
        L=385,  # 剪切长度
        num_workers=128,  # 可根据机器调整
    )
