import argparse
import csv
import math
from pathlib import Path

import torch
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from dataset import BridgeDataset, bridge_collate_fn
from evaluate import build_seg_metric_from_cfg, flatten_seg_results, update_seg_metric, validate
from losses import build_loss_from_cfg
from model import build_model_from_cfg, get_task_info
from utils import get_lr, load_yaml, move_batch_to_device, save_checkpoint, set_seed

import torch
from torch.cuda.amp import autocast


class AverageMeter:

    def __init__(self, device):
        self.device = device
        self.reset()

    def reset(self):

        self.sum = None
        self.count = 0

    def update(
        self,
        value,
        n=1,
    ):

        n = int(n)

        if n <= 0:
            return

        if torch.is_tensor(value):

            value = value.detach()

            if value.numel() != 1:
                value = value.mean()

            value = value.to(
                device=self.device,
                dtype=torch.float32,
            )

        else:

            value = torch.tensor(
                float(value),
                device=self.device,
                dtype=torch.float32,
            )

        weighted_value = (
            value * float(n)
        )

        if self.sum is None:
            self.sum = weighted_value
        else:
            self.sum = (
                self.sum
                + weighted_value
            )

        self.count += n

    @property
    def avg(self):

        if (
            self.sum is None
            or self.count == 0
        ):
            return 0.0

        return float(
            (
                self.sum
                / float(self.count)
            )
            .detach()
            .cpu()
            .item()
        )

def count_valid_samples(
    flags,
    batch_size,
):

    if flags is None:
        return int(batch_size)

    if torch.is_tensor(flags):

        return int(
            flags
            .detach()
            .bool()
            .sum()
            .item()
        )

    return sum(
        1
        for flag in flags
        if bool(flag)
    )

def train_one_epoch(
    model,
    dataloader,
    criterion,
    optimizer,
    device,
    epoch,
    cfg,
    scaler=None,
    ema=None,
    logger=None,
    writer=None,
):

    model.train()

    train_cfg = cfg.get(
        "train",
        {},
    )

    task_type = cfg.get(
        "task",
        {},
    ).get(
        "type",
        "multitask",
    )

    task_info = get_task_info(
        cfg
    )

    amp_enabled = (
        bool(
            train_cfg.get(
                "amp",
                True,
            )
        )
        and device.type == "cuda"
    )

    log_interval = max(
        1,
        int(
            train_cfg.get(
                "log_interval",
                20,
            )
        ),
    )

    grad_accum_steps = max(
        1,
        int(
            train_cfg.get(
                "grad_accum_steps",
                1,
            )
        ),
    )

    loss_meter = AverageMeter(
        device
    )

    seg_loss_meter = AverageMeter(
        device
    )

    ce_loss_meter = AverageMeter(
        device
    )

    dice_loss_meter = AverageMeter(
        device
    )

    det_loss_meter = AverageMeter(
        device
    )

    det_cls_loss_meter = AverageMeter(
        device
    )

    det_box_loss_meter = AverageMeter(
        device
    )

    det_ctr_loss_meter = AverageMeter(
        device
    )

    det_pos_meter = AverageMeter(
        device
    )

    seg_sample_count = 0
    det_sample_count = 0

    compute_seg_metric = bool(
        train_cfg.get(
            "compute_train_seg_metric",
            True,
        )
    )

    use_seg_metric = (
        task_type
        in [
            "segmentation",
            "multitask",
        ]
        and compute_seg_metric
    )

    if use_seg_metric:

        seg_metric = (
            build_seg_metric_from_cfg(
                cfg
            )
        )

    else:

        seg_metric = None

    num_batches = len(
        dataloader
    )

    optimizer.zero_grad(
        set_to_none=True
    )

    for step, batch in enumerate(
        dataloader,
        start=1,
    ):

        batch = move_batch_to_device(
            batch,
            device,
        )

        images = batch[
            "images"
        ]

        batch_size = images.shape[
            0
        ]

        with autocast(
            enabled=amp_enabled
        ):

            outputs = model(
                images
            )

            loss_dict = criterion(
                outputs,
                batch,
            )

            loss = loss_dict[
                "loss"
            ]

            loss_for_backward = (
                loss
                / grad_accum_steps
            )

        if (
            scaler is not None
            and amp_enabled
        ):

            scaler.scale(
                loss_for_backward
            ).backward()

        else:

            loss_for_backward.backward()

        should_step = (
            step % grad_accum_steps == 0
            or step == num_batches
        )

        if should_step:

            if (
                scaler is not None
                and amp_enabled
            ):

                scaler.step(
                    optimizer
                )

                scaler.update()

            else:

                optimizer.step()

            optimizer.zero_grad(
                set_to_none=True
            )

            if ema is not None:

                ema.update(
                    model
                )

        seg_count = count_valid_samples(
            batch.get(
                "has_seg",
                None,
            ),
            batch_size,
        )

        det_count = count_valid_samples(
            batch.get(
                "has_det",
                None,
            ),
            batch_size,
        )

        seg_sample_count += (
            seg_count
        )

        det_sample_count += (
            det_count
        )

        loss_meter.update(
            loss,
            n=batch_size,
        )

        if seg_count > 0:

            seg_loss_meter.update(
                loss_dict.get(
                    "loss_seg",
                    0.0,
                ),
                n=seg_count,
            )

            ce_loss_meter.update(
                loss_dict.get(
                    "loss_ce",
                    0.0,
                ),
                n=seg_count,
            )

            dice_loss_meter.update(
                loss_dict.get(
                    "loss_dice",
                    0.0,
                ),
                n=seg_count,
            )

        if det_count > 0:

            det_loss_meter.update(
                loss_dict.get(
                    "loss_det",
                    0.0,
                ),
                n=det_count,
            )

            det_cls_loss_meter.update(
                loss_dict.get(
                    "loss_det_cls",
                    0.0,
                ),
                n=det_count,
            )

            det_box_loss_meter.update(
                loss_dict.get(
                    "loss_det_box",
                    0.0,
                ),
                n=det_count,
            )

            det_ctr_loss_meter.update(
                loss_dict.get(
                    "loss_det_ctr",
                    0.0,
                ),
                n=det_count,
            )

            num_det_pos = int(
                loss_dict.get(
                    "num_det_pos",
                    0,
                )
            )

            det_pos_meter.update(
                num_det_pos
                / max(
                    det_count,
                    1,
                ),
                n=det_count,
            )

        if seg_metric is not None:

            update_seg_metric(
                seg_metric,
                outputs,
                batch,
            )

        if (
            logger is not None
            and (
                step % log_interval == 0
                or step == num_batches
            )
        ):

            lr = get_lr(
                optimizer
            )

            logger.info(
                f"[Train E{epoch:03d}] "
                f"[{step:04d}/{num_batches:04d}] "
                f"loss={loss_meter.avg:.4f} | "
                f"seg={seg_loss_meter.avg:.4f} | "
                f"det={det_loss_meter.avg:.4f} | "
                f"seg_n={seg_count} | "
                f"det_n={det_count} | "
                f"det_pos/img={det_pos_meter.avg:.2f} | "
                f"lr={lr:.6e}"
            )

    result = {

        "train_loss":
            loss_meter.avg,

        "train_loss_seg":
            seg_loss_meter.avg,

        "train_loss_ce":
            ce_loss_meter.avg,

        "train_loss_dice":
            dice_loss_meter.avg,

        "train_loss_det":
            det_loss_meter.avg,

        "train_loss_det_cls":
            det_cls_loss_meter.avg,

        "train_loss_det_box":
            det_box_loss_meter.avg,

        "train_loss_det_ctr":
            det_ctr_loss_meter.avg,

        "train_det_pos_per_img":
            det_pos_meter.avg,

        "train_num_seg_samples":
            int(seg_sample_count),

        "train_num_det_samples":
            int(det_sample_count),

        "lr":
            get_lr(optimizer),
    }

    if seg_metric is not None:

        seg_result = (
            seg_metric.compute()
        )

        result.update(
            flatten_seg_results(
                seg_result=seg_result,
                class_names=(
                    task_info.seg_class_names
                ),
                prefix="train_seg",
            )
        )

        if logger is not None:

            logger.info(
                f"[Train E{epoch:03d}] "
                f"[SEG] "
                f"{seg_metric.summary_string()}"
            )

            logger.info(
                f"[Train E{epoch:03d}] "
                f"[SEG-CLS] "
                f"{seg_metric.class_summary_string()}"
            )

    if writer is not None:

        writer.add_scalar(
            "Train/Loss",
            result["train_loss"],
            epoch,
        )

        writer.add_scalar(
            "Train/LossSeg",
            result["train_loss_seg"],
            epoch,
        )

        writer.add_scalar(
            "Train/LossCE",
            result["train_loss_ce"],
            epoch,
        )

        writer.add_scalar(
            "Train/LossDice",
            result["train_loss_dice"],
            epoch,
        )

        writer.add_scalar(
            "Train/LossDet",
            result["train_loss_det"],
            epoch,
        )

        writer.add_scalar(
            "Train/LossDetCls",
            result["train_loss_det_cls"],
            epoch,
        )

        writer.add_scalar(
            "Train/LossDetBox",
            result["train_loss_det_box"],
            epoch,
        )

        writer.add_scalar(
            "Train/LossDetCtr",
            result["train_loss_det_ctr"],
            epoch,
        )

        writer.add_scalar(
            "Train/LR",
            result["lr"],
            epoch,
        )

        if "train_seg_mIoU" in result:

            writer.add_scalar(
                "TrainSeg/mIoU",
                result["train_seg_mIoU"],
                epoch,
            )

            writer.add_scalar(
                "TrainSeg/mDice",
                result["train_seg_mDice"],
                epoch,
            )

            writer.add_scalar(
                "TrainSeg/PixelAcc",
                result["train_seg_PixelAcc"],
                epoch,
            )

    return result

def build_loader(cfg, split, train=False):
    data = cfg['data']
    task = get_task_info(cfg)
    ds = BridgeDataset(
        image_dir=data[f'{split}_image_dir'],
        mask_dir=data.get(f'{split}_mask_dir'),
        label_dir=data.get(f'{split}_label_dir'),
        img_size=int(data.get('img_size', 512)),
        is_train=train,
        aug_level=data.get('aug_level', 'none') if train else 'none',
        normalize_mode=data.get('normalize_mode', 'zero_one'),
        mask_encoding=data.get('mask_encoding', 'index'),
        task_type=cfg.get('task', {}).get('type', 'multitask'),
        seg_num_classes=task.seg_num_classes,
        det_num_classes=task.det_num_classes,
        seg_class_names=task.seg_class_names,
        det_class_names=task.det_class_names,
    )
    return DataLoader(
        ds,
        batch_size=int(cfg['train'].get('batch_size', 1)),
        shuffle=train,
        num_workers=int(data.get('num_workers', 4)),
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        collate_fn=bridge_collate_fn,
    )

def build_scheduler(cfg, optimizer):
    tc = cfg['train']
    epochs = int(tc.get('epochs', 120))
    warmup = int(tc.get('warmup_epochs', 0))
    base_lr = float(tc.get('lr', 1e-4))
    min_lr = float(tc.get('min_lr', 1e-6))
    min_factor = min(max(min_lr / max(base_lr, 1e-12), 0.0), 1.0)

    def schedule(epoch):
        if warmup > 0 and epoch < warmup:
            return max(min_factor, (epoch + 1) / warmup)
        progress = min(max((epoch - warmup) / max(1, epochs - warmup), 0.0), 1.0)
        return min_factor + (1.0 - min_factor) * 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)

def write_csv(path, row):
    path = Path(path)
    values = {}
    for key, value in row.items():
        if torch.is_tensor(value):
            value = value.detach().cpu().item() if value.numel() == 1 else value.detach().cpu().tolist()
        values[key] = value
    exists = path.exists()
    with path.open('a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(values.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(values)

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', default='configs/mtl.yaml')
    p.add_argument('--device', default=None)
    p.add_argument('--seed', type=int, default=None)
    p.add_argument('--name', default=None)
    args = p.parse_args()

    cfg = load_yaml(args.config)
    seed = int(args.seed if args.seed is not None else cfg['train'].get('seed', 42))
    cfg['train']['seed'] = seed
    set_seed(seed)

    device = torch.device(args.device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    run_root = Path(cfg.get('project', {}).get('run_root', 'runs/mtl'))
    base_name = args.name or cfg.get('project', {}).get('run_name', 'dp2lnet')
    run_dir = run_root / f'{base_name}_seed{seed}'
    ckpt_dir = run_dir / 'checkpoints'
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    train_loader = build_loader(cfg, 'train', True)
    val_loader = build_loader(cfg, 'val', False)
    model = build_model_from_cfg(cfg).to(device)
    criterion = build_loss_from_cfg(cfg).to(device)

    tc = cfg['train']
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(tc.get('lr', 1e-4)),
        weight_decay=float(tc.get('weight_decay', 1e-4)),
    )
    scheduler = build_scheduler(cfg, optimizer)
    scaler = GradScaler(enabled=bool(tc.get('amp', True)) and device.type == 'cuda')

    metric_cfg = cfg.get('metric', {})
    w_seg = float(metric_cfg.get('score_weight_seg', 0.5))
    w_det = float(metric_cfg.get('score_weight_det', 0.5))
    best_score = -1.0
    epochs = int(tc.get('epochs', 120))

    print(f'device={device} seed={seed} train={len(train_loader.dataset)} val={len(val_loader.dataset)}')

    for epoch in range(1, epochs + 1):
        train_result = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            cfg=cfg,
            scaler=scaler,
            ema=None,
            logger=None,
            writer=None,
        )
        val_result = validate(
            model=model,
            dataloader=val_loader,
            device=device,
            cfg=cfg,
            criterion=criterion,
            epoch=epoch,
            logger=None,
            writer=None,
        )
        scheduler.step()

        score = (
            w_seg * float(val_result['val_seg_mIoU'])
            + w_det * float(val_result['val_det_mAP50'])
        )
        row = {'epoch': epoch, **train_result, **val_result, 'val_score': score}
        write_csv(run_dir / 'results.csv', row)

        save_checkpoint(
            ckpt_dir / 'last.pth', model, optimizer, scheduler, scaler,
            epoch=epoch, best_metric=max(best_score, score), cfg=cfg,
        )
        if score > best_score:
            best_score = score
            save_checkpoint(
                ckpt_dir / 'best.pth', model, optimizer, scheduler, scaler,
                epoch=epoch, best_metric=best_score, cfg=cfg,
            )

        print(
            f'epoch {epoch:03d}/{epochs} '
            f'loss={float(train_result.get("train_loss", 0.0)):.4f} '
            f'mIoU={float(val_result["val_seg_mIoU"]):.4f} '
            f'mAP50={float(val_result["val_det_mAP50"]):.4f} '
            f'score={score:.4f}'
        )

if __name__ == '__main__':
    main()
