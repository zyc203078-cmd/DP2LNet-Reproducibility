import argparse
import json
from pathlib import Path

import torch
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader

from dataset import BridgeDataset, bridge_collate_fn
from model import build_model_from_cfg, get_task_info
from metrics import SegmentationMetric, DetectionMetric
from utils import AverageMeter, load_yaml, move_batch_to_device

from typing import Dict, List

import torch


def build_seg_metric_from_cfg(cfg):

    task_info = get_task_info(cfg)

    metric = SegmentationMetric(
        num_classes=task_info.seg_num_classes,
        class_names=task_info.seg_class_names,
        ignore_index=None,
    )

    return metric

def build_det_metric_from_cfg(cfg):

    task_info = get_task_info(cfg)
    eval_cfg = cfg.get("eval", {})

    metric = DetectionMetric(
        num_classes=task_info.det_num_classes,
        class_names=task_info.det_class_names,

        conf_thres=float(
            eval_cfg.get(
                "conf_thres",
                0.25,
            )
        ),

        pr_iou_thres=float(
            eval_cfg.get(
                "pr_iou_thres",
                0.50,
            )
        ),

        map_iou_start=float(
            eval_cfg.get(
                "map_iou_start",
                0.50,
            )
        ),

        map_iou_end=float(
            eval_cfg.get(
                "map_iou_end",
                0.95,
            )
        ),

        map_iou_step=float(
            eval_cfg.get(
                "map_iou_step",
                0.05,
            )
        ),
    )

    return metric

def get_valid_indices(
    flags,
    batch_size: int,
) -> List[int]:

    if flags is None:

        return list(
            range(batch_size)
        )

    if torch.is_tensor(flags):

        flags = (
            flags
            .detach()
            .cpu()
            .tolist()
        )

    if len(flags) != batch_size:

        raise ValueError(
            f"has_seg / has_det 长度错误："
            f"len={len(flags)}, "
            f"batch_size={batch_size}"
        )

    return [
        index
        for index, flag in enumerate(flags)
        if bool(flag)
    ]

def select_det_outputs_by_indices(
    det_outputs: Dict,
    indices: List[int],
) -> Dict:

    if len(indices) == 0:

        raise ValueError(
            "检测样本 indices 不能为空。"
        )

    selected = {}

    batch_keys = {
        "cls_logits",
        "box_reg",
        "centerness",
    }

    for key, value in det_outputs.items():

        if key in batch_keys:

            if not isinstance(
                value,
                (list, tuple),
            ):

                raise TypeError(
                    f"det_outputs['{key}'] "
                    f"必须为 list / tuple。"
                )

            selected[key] = [
                feature[indices]
                for feature in value
            ]

        else:

            selected[key] = value

    return selected

@torch.no_grad()
def update_seg_metric(
    metric,
    outputs: Dict,
    batch: Dict,
):

    if outputs.get(
        "seg_logits",
        None,
    ) is None:

        return

    if "masks" not in batch:

        raise KeyError(
            "batch 中缺少 masks。"
        )

    seg_logits = outputs[
        "seg_logits"
    ]

    masks = batch[
        "masks"
    ]

    if seg_logits.ndim != 4:

        raise ValueError(
            f"seg_logits 应为 [B,C,H,W]，"
            f"当前 shape={seg_logits.shape}"
        )

    if masks.ndim != 3:

        raise ValueError(
            f"masks 应为 [B,H,W]，"
            f"当前 shape={masks.shape}"
        )

    batch_size = (
        seg_logits.shape[0]
    )

    if masks.shape[0] != batch_size:

        raise ValueError(
            "seg_logits 与 masks "
            "batch size 不一致。"
        )

    seg_indices = get_valid_indices(
        batch.get(
            "has_seg",
            None,
        ),
        batch_size,
    )

    if len(seg_indices) == 0:
        return

    seg_logits = seg_logits[
        seg_indices
    ]

    masks = masks[
        seg_indices
    ]

    pred = torch.argmax(
        seg_logits,
        dim=1,
    )

    metric.update(
        pred,
        masks,
    )

def get_det_head(model):

    module = (
        model.module
        if hasattr(model, "module")
        else model
    )

    if not hasattr(
        module,
        "det_head",
    ):

        raise AttributeError(
            "当前模型中不存在 det_head。"
        )

    return module.det_head

@torch.no_grad()
def update_det_metric(
    metric,
    model,
    outputs: Dict,
    batch: Dict,
    cfg,
):

    if outputs.get(
        "det_outputs",
        None,
    ) is None:

        return

    if "boxes" not in batch:

        raise KeyError(
            "batch 中缺少 boxes。"
        )

    if "labels" not in batch:

        raise KeyError(
            "batch 中缺少 labels。"
        )

    if "images" not in batch:

        raise KeyError(
            "batch 中缺少 images。"
        )

    images = batch[
        "images"
    ]

    batch_size = (
        images.shape[0]
    )

    det_indices = get_valid_indices(
        batch.get(
            "has_det",
            None,
        ),
        batch_size,
    )

    if len(det_indices) == 0:
        return

    det_outputs = (
        select_det_outputs_by_indices(
            outputs["det_outputs"],
            det_indices,
        )
    )

    det_head = get_det_head(
        model
    )

    eval_cfg = cfg.get(
        "eval",
        {},
    )

    img_size = (
        images.shape[-2],
        images.shape[-1],
    )

    preds = det_head.decode_predictions(
        det_outputs=det_outputs,
        img_size=img_size,

        conf_thres=0.001,

        nms_iou=float(
            eval_cfg.get(
                "nms_iou",
                0.50,
            )
        ),

        max_detections=int(
            eval_cfg.get(
                "max_detections",
                300,
            )
        ),
    )

    targets = []

    for index in det_indices:

        targets.append(
            {
                "boxes":
                    batch["boxes"][index],

                "labels":
                    batch["labels"][index],
            }
        )

    if len(preds) != len(targets):

        raise ValueError(
            f"检测预测数量与 GT 数量不一致："
            f"pred={len(preds)}, "
            f"target={len(targets)}"
        )

    metric.update(
        preds,
        targets,
    )

def flatten_seg_results(
    seg_result: Dict,
    class_names: List[str],
    prefix: str = "seg",
) -> Dict:

    out = {
        f"{prefix}_mIoU":
            float(
                seg_result["mIoU"]
            ),

        f"{prefix}_mDice":
            float(
                seg_result["mDice"]
            ),

        f"{prefix}_PixelAcc":
            float(
                seg_result["PixelAcc"]
            ),

        f"{prefix}_mPrecision":
            float(
                seg_result.get(
                    "mPrecision",
                    0.0,
                )
            ),

        f"{prefix}_mRecall":
            float(
                seg_result.get(
                    "mRecall",
                    0.0,
                )
            ),
    }

    ious = seg_result[
        "IoU_per_class"
    ]

    dices = seg_result[
        "Dice_per_class"
    ]

    precisions = seg_result.get(
        "Precision_per_class",
        [0.0] * len(class_names),
    )

    recalls = seg_result.get(
        "Recall_per_class",
        [0.0] * len(class_names),
    )

    if len(ious) != len(class_names):

        raise ValueError(
            "IoU_per_class 与 "
            "class_names 数量不一致。"
        )

    if len(dices) != len(class_names):

        raise ValueError(
            "Dice_per_class 与 "
            "class_names 数量不一致。"
        )

    if len(ious) > 1:

        foreground_ious = [
            float(value)
            for value in ious[1:]
        ]

        out[
            f"{prefix}_mIoU_fg"
        ] = (
            sum(foreground_ious)
            / len(foreground_ious)
        )

    for (
        class_name,
        iou,
        dice,
        precision,
        recall,
    ) in zip(
        class_names,
        ious,
        dices,
        precisions,
        recalls,
    ):

        out[
            f"{prefix}_IoU_{class_name}"
        ] = float(iou)

        out[
            f"{prefix}_Dice_{class_name}"
        ] = float(dice)

        out[
            f"{prefix}_Precision_{class_name}"
        ] = float(precision)

        out[
            f"{prefix}_Recall_{class_name}"
        ] = float(recall)

    return out

def get_per_class_result(
    per_class,
    class_id: int,
) -> Dict:

    default = {
        "P": 0.0,
        "R": 0.0,
        "F1": 0.0,
        "mAP50": 0.0,
        "mAP50_95": 0.0,
    }

    if per_class is None:
        return default

    if isinstance(
        per_class,
        list,
    ):

        if class_id < len(
            per_class
        ):

            return per_class[
                class_id
            ]

        return default

    if isinstance(
        per_class,
        dict,
    ):

        if class_id in per_class:

            return per_class[
                class_id
            ]

        if str(class_id) in per_class:

            return per_class[
                str(class_id)
            ]

    return default

def flatten_det_results(
    det_result: Dict,
    class_names: List[str],
    prefix: str = "det",
) -> Dict:

    out = {
        f"{prefix}_P":
            float(
                det_result["P"]
            ),

        f"{prefix}_R":
            float(
                det_result["R"]
            ),

        f"{prefix}_F1":
            float(
                det_result["F1"]
            ),

        f"{prefix}_mAP50":
            float(
                det_result["mAP50"]
            ),

        f"{prefix}_mAP50_95":
            float(
                det_result["mAP50_95"]
            ),
    }

    per_class = det_result.get(
        "per_class",
        None,
    )

    for class_id, class_name in enumerate(
        class_names
    ):

        class_result = (
            get_per_class_result(
                per_class,
                class_id,
            )
        )

        out[
            f"{prefix}_P_{class_name}"
        ] = float(
            class_result.get(
                "P",
                0.0,
            )
        )

        out[
            f"{prefix}_R_{class_name}"
        ] = float(
            class_result.get(
                "R",
                0.0,
            )
        )

        out[
            f"{prefix}_F1_{class_name}"
        ] = float(
            class_result.get(
                "F1",
                0.0,
            )
        )

        out[
            f"{prefix}_mAP50_{class_name}"
        ] = float(
            class_result.get(
                "mAP50",
                0.0,
            )
        )

        out[
            f"{prefix}_mAP50_95_{class_name}"
        ] = float(
            class_result.get(
                "mAP50_95",
                0.0,
            )
        )

    return out

import torch
from torch.cuda.amp import autocast


def count_valid_samples(
    flags,
    batch_size: int,
) -> int:

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

def scalar_to_float(
    value,
) -> float:

    if torch.is_tensor(value):

        return float(
            value.detach().item()
        )

    return float(
        value
    )

@torch.no_grad()
def validate(
    model,
    dataloader,
    device,
    cfg,
    criterion=None,
    epoch=0,
    logger=None,
    writer=None,
):

    model.eval()

    train_cfg = cfg.get(
        "train",
        {},
    )

    task_cfg = cfg.get(
        "task",
        {},
    )

    task_info = get_task_info(
        cfg
    )

    task_type = task_cfg.get(
        "type",
        "multitask",
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

    use_seg = task_type in [
        "segmentation",
        "multitask",
    ]

    use_det = task_type in [
        "detection",
        "multitask",
    ]

    if use_seg:

        seg_metric = (
            build_seg_metric_from_cfg(
                cfg
            )
        )

    else:

        seg_metric = None

    if use_det:

        det_metric = (
            build_det_metric_from_cfg(
                cfg
            )
        )

    else:

        det_metric = None

    loss_meter = AverageMeter()

    seg_loss_meter = AverageMeter()

    det_loss_meter = AverageMeter()

    seg_sample_count = 0
    det_sample_count = 0

    for batch in dataloader:

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

            if criterion is not None:

                loss_dict = criterion(
                    outputs,
                    batch,
                )

                loss = loss_dict[
                    "loss"
                ]

            else:

                loss_dict = None
                loss = None

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

        if loss is not None:

            loss_meter.update(
                scalar_to_float(
                    loss
                ),
                n=batch_size,
            )

            if seg_count > 0:

                seg_loss_meter.update(
                    scalar_to_float(
                        loss_dict.get(
                            "loss_seg",
                            0.0,
                        )
                    ),
                    n=seg_count,
                )

            if det_count > 0:

                det_loss_meter.update(
                    scalar_to_float(
                        loss_dict.get(
                            "loss_det",
                            0.0,
                        )
                    ),
                    n=det_count,
                )

        if seg_metric is not None:

            update_seg_metric(
                seg_metric,
                outputs,
                batch,
            )

        if det_metric is not None:

            update_det_metric(
                det_metric,
                model,
                outputs,
                batch,
                cfg,
            )

    result = {

        "val_loss":
            loss_meter.avg,

        "val_loss_seg":
            seg_loss_meter.avg,

        "val_loss_det":
            det_loss_meter.avg,

        "val_num_seg_samples":
            int(seg_sample_count),

        "val_num_det_samples":
            int(det_sample_count),
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
                prefix="val_seg",
            )
        )

        if logger is not None:

            logger.info(
                f"[VAL E{epoch:03d}] "
                f"[SEG] "
                f"{seg_metric.summary_string()}"
            )

            logger.info(
                f"[VAL E{epoch:03d}] "
                f"[SEG-CLS] "
                f"{seg_metric.class_summary_string()}"
            )

    if det_metric is not None:

        det_result = (
            det_metric.compute()
        )

        result.update(
            flatten_det_results(
                det_result=det_result,
                class_names=(
                    task_info.det_class_names
                ),
                prefix="val_det",
            )
        )

        if logger is not None:

            logger.info(
                f"[VAL E{epoch:03d}] "
                f"[DET] "
                f"{det_metric.summary_string()}"
            )

            logger.info(
                f"[VAL E{epoch:03d}] "
                f"[DET-CLS] "
                f"{det_metric.class_summary_string()}"
            )

    if writer is not None:

        writer.add_scalar(
            "Val/Loss",
            result["val_loss"],
            epoch,
        )

        writer.add_scalar(
            "Val/LossSeg",
            result["val_loss_seg"],
            epoch,
        )

        writer.add_scalar(
            "Val/LossDet",
            result["val_loss_det"],
            epoch,
        )

        if "val_seg_mIoU" in result:

            writer.add_scalar(
                "ValSeg/mIoU",
                result[
                    "val_seg_mIoU"
                ],
                epoch,
            )

            writer.add_scalar(
                "ValSeg/mDice",
                result[
                    "val_seg_mDice"
                ],
                epoch,
            )

            writer.add_scalar(
                "ValSeg/PixelAcc",
                result[
                    "val_seg_PixelAcc"
                ],
                epoch,
            )

        if "val_det_mAP50" in result:

            writer.add_scalar(
                "ValDet/P",
                result[
                    "val_det_P"
                ],
                epoch,
            )

            writer.add_scalar(
                "ValDet/R",
                result[
                    "val_det_R"
                ],
                epoch,
            )

            writer.add_scalar(
                "ValDet/F1",
                result[
                    "val_det_F1"
                ],
                epoch,
            )

            writer.add_scalar(
                "ValDet/mAP50",
                result[
                    "val_det_mAP50"
                ],
                epoch,
            )

            writer.add_scalar(
                "ValDet/mAP50_95",
                result[
                    "val_det_mAP50_95"
                ],
                epoch,
            )

    return result


def build_loader(cfg, split):
    data = cfg["data"]
    task = get_task_info(cfg)
    ds = BridgeDataset(
        image_dir=data[f"{split}_image_dir"],
        mask_dir=data.get(f"{split}_mask_dir"),
        label_dir=data.get(f"{split}_label_dir"),
        img_size=int(data.get("img_size", 512)),
        is_train=False,
        aug_level="none",
        normalize_mode=data.get("normalize_mode", "zero_one"),
        mask_encoding=data.get("mask_encoding", "index"),
        task_type=cfg.get("task", {}).get("type", "multitask"),
        seg_num_classes=task.seg_num_classes,
        det_num_classes=task.det_num_classes,
        seg_class_names=task.seg_class_names,
        det_class_names=task.det_class_names,
    )
    return DataLoader(
        ds,
        batch_size=int(cfg.get("train", {}).get("batch_size", 1)),
        shuffle=False,
        num_workers=int(data.get("num_workers", 4)),
        pin_memory=torch.cuda.is_available(),
        collate_fn=bridge_collate_fn,
    )

def load_weights(model, path, device):
    ckpt = torch.load(path, map_location=device)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state, strict=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--device", default=None)
    parser.add_argument("--output", default="test_metrics.json")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    loader = build_loader(cfg, args.split)
    model = build_model_from_cfg(cfg).to(device)
    load_weights(model, args.weights, device)
    result = validate(model, loader, device, cfg, criterion=None, epoch=0)

    for key, value in result.items():
        print(f"{key}: {value}")
    Path(args.output).write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

if __name__ == "__main__":
    main()
