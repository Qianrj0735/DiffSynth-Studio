from huggingface_hub import snapshot_download

# 下载最新版本的仓库到默认缓存
local_path = snapshot_download(repo_id="alibaba-pai/Wan2.1-Fun-1.3B-InP")
print(f"模型已下载至：{local_path}")
