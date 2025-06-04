import yaml


def load_config(yaml_path: str) -> dict:
    """
    读取 YAML 文件并返回一个字典：
      {
        'block1': { 'mode': 'm', 'tensors': [ {name, index:{...}}, ... ] },
        'block2': { ... },
        ...
      }
    """
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict) or "blocks" not in data:
        raise ValueError("YAML 中缺少顶层 'blocks' 字段。")
    return data["blocks"]


def validate_block(block_name: str, block_info: dict):
    """
    校验某个 block 中 mode 和 tensors 数量是否匹配：
      m -> tensors 长度 == 1
      s -> tensors 长度 == 4
      xs -> tensors 长度 == 16
    """
    mode = block_info.get("mode")
    tensors = block_info.get("tensors")
    if mode not in ("m", "s", "xs"):
        raise ValueError(
            f"Block {block_name} 的 mode 必须是 'm'、's' 或 'xs'，当前是：{mode}"
        )
    if not isinstance(tensors, list):
        raise ValueError(f"Block {block_name} 的 'tensors' 应该是列表。")
    expected_len = {"m": 1, "s": 4, "xs": 16}[mode]
    if len(tensors) != expected_len:
        raise ValueError(
            f"Block {block_name} 的 mode='{mode}' 应该有 {expected_len} 个 tensor，"
            f"但实际写了 {len(tensors)} 个。"
        )


def compute_actual_indices(blocks_cfg: dict, x: int) -> dict:
    """
    根据 blocks_cfg 和给定的 DataLoader 索引 x，
    返回一个字典：{ block_name: [ (tensor_name, 实际帧索引), ... ], ... }
    """
    result = {}
    for block_name, block_info in blocks_cfg.items():
        validate_block(block_name, block_info)
        mode = block_info["mode"]
        tensors = block_info["tensors"]
        actual_list = []

        for tensor_def in tensors:
            name = tensor_def.get("name")
            if name is None:
                raise ValueError(f"Block {block_name} 中有某个 tensor 缺少 'name'。")

            idx_cfg = tensor_def.get("index")
            if not isinstance(idx_cfg, dict):
                raise ValueError(
                    f"Block {block_name}, tensor {name} 的 'index' 应该是 dict。"
                )

            # 判断是 rel_offset 还是 fixed
            if "rel_offset" in idx_cfg and "fixed" in idx_cfg:
                raise ValueError(
                    f"Block {block_name}, tensor {name} 的 'index' 同时出现 rel_offset 和 fixed，二选一即可。"
                )
            if "rel_offset" in idx_cfg:
                offset = idx_cfg["rel_offset"]
                if not isinstance(offset, int):
                    raise ValueError(
                        f"Block {block_name}, tensor {name} 的 rel_offset 必须是整数。"
                    )
                actual_idx = x + offset
            elif "fixed" in idx_cfg:
                fixed = idx_cfg["fixed"]
                if not isinstance(fixed, int):
                    raise ValueError(
                        f"Block {block_name}, tensor {name} 的 fixed 必须是整数。"
                    )
                actual_idx = fixed
            else:
                raise ValueError(
                    f"Block {block_name}, tensor {name} 的 'index' 内必须包含 rel_offset 或 fixed。"
                )

            actual_list.append(int(actual_idx))

        result[block_name] = actual_list

    return result





if __name__ == "__main__":
    # 假设你的 YAML 文件名为 config.yaml，且放在当前目录
    yaml_path = "examples/wanvideo/predict_pack_configs/9hrz45.yml"
    blocks_cfg = load_config(yaml_path)

    # 举例：当 DataLoader 给的 x=10 时
    x = 10
    actual_indices_dict = compute_actual_indices(blocks_cfg, x)

    # 打印每个 block 下，每个 tensor 对应的“真实”帧索引
    for block_name, tensor_list in actual_indices_dict.items():
        print(f"=== {block_name}（mode={blocks_cfg[block_name]['mode']}） ===")
        for tensor_name, actual_idx in tensor_list:
            print(f"  • Tensor '{tensor_name}'  ->  实际读取帧索引 = {actual_idx}")
        print()
