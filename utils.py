from pathlib import Path

import torch
import yaml

def ensure_dir(path):

    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path

def load_yaml(path):

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"配置文件不存在：{path}")

    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if not isinstance(cfg, dict):
        raise ValueError(f"YAML 配置内容必须为字典：{path}")

    return cfg

def save_yaml(cfg, path):

    path = Path(path)
    ensure_dir(path.parent)

    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)

class AverageMeter:

    def __init__(self):
        self.reset()

    def reset(self):

        self.val = 0.0
        self.sum = 0.0
        self.count = 0
        self.avg = 0.0

    def update(self, val, n: int = 1):

        if torch.is_tensor(val):
            if val.numel() != 1:
                val = val.mean()
            val = float(val.detach().cpu().item())
        else:
            val = float(val)

        n = int(n)

        if n <= 0:
            return

        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def count_parameters(model):

    total_params = sum(param.numel() for param in model.parameters())
    trainable_params = sum(param.numel() for param in model.parameters() if param.requires_grad)

    return total_params, trainable_params

def format_num_params(num_params):

    return f"{num_params / 1e6:.2f}M"

def move_batch_to_device(batch, device):

    new_batch = {}

    for key, value in batch.items():

        if torch.is_tensor(value):
            new_batch[key] = value.to(device, non_blocking=True)

        elif isinstance(value, list):
            new_batch[key] = [
                item.to(device, non_blocking=True) if torch.is_tensor(item) else item
                for item in value
            ]

        else:
            new_batch[key] = value

    return new_batch

def get_lr(optimizer):

    return float(optimizer.param_groups[0]["lr"])


import os
import random

import numpy as np
import torch

def set_seed(seed: int = 42, deterministic: bool = False):

    seed = int(seed)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    os.environ["PYTHONHASHSEED"] = str(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


from pathlib import Path

import torch


def save_checkpoint(
    save_path,
    model,
    optimizer=None,
    scheduler=None,
    scaler=None,
    ema=None,
    epoch: int = 0,
    best_metric=None,
    cfg=None,
):

    save_path = Path(save_path)
    ensure_dir(save_path.parent)

    state = {
        "epoch": int(epoch),
        "model": model.state_dict(),
        "best_metric": best_metric,
        "cfg": cfg,
    }

    if optimizer is not None:
        state["optimizer"] = optimizer.state_dict()

    if scheduler is not None:
        state["scheduler"] = scheduler.state_dict()

    if scaler is not None:
        state["scaler"] = scaler.state_dict()

    if ema is not None:
        state["ema"] = ema.ema.state_dict()

    torch.save(state, save_path)

def load_checkpoint(
    checkpoint_path,
    model,
    optimizer=None,
    scheduler=None,
    scaler=None,
    ema=None,
    map_location="cpu",
    strict: bool = True,
):

    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint 不存在：{checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location=map_location)

    if not isinstance(ckpt, dict) or "model" not in ckpt:
        raise ValueError(
            f"Checkpoint 格式错误：{checkpoint_path}\n"
            "当前训练恢复要求 checkpoint 中包含 'model' 字段。"
        )

    model.load_state_dict(ckpt["model"], strict=strict)

    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])

    if scheduler is not None and "scheduler" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler"])

    if scaler is not None and "scaler" in ckpt:
        scaler.load_state_dict(ckpt["scaler"])

    if ema is not None and "ema" in ckpt:
        ema.ema.load_state_dict(ckpt["ema"], strict=strict)

    return ckpt
