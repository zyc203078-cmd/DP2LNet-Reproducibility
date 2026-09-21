from model import get_task_info


import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):

    def __init__(self, num_classes):
        super().__init__()

        self.num_classes = int(num_classes)

    def forward(self, logits, target):

        probs = torch.softmax(
            logits,
            dim=1,
        )

        target_onehot = F.one_hot(
            target,
            num_classes=self.num_classes,
        )

        target_onehot = target_onehot.permute(
            0,
            3,
            1,
            2,
        ).float()

        probs = probs[:, 1:]
        target_onehot = target_onehot[:, 1:]

        intersection = torch.sum(
            probs * target_onehot,
            dim=(0, 2, 3),
        )

        denominator = torch.sum(
            probs + target_onehot,
            dim=(0, 2, 3),
        )

        dice = (
            2.0 * intersection + 1.0
        ) / (
            denominator + 1.0
        )

        loss = 1.0 - dice.mean()

        return loss

class SegLoss(nn.Module):

    def __init__(self, num_classes):
        super().__init__()

        self.ce_loss = nn.CrossEntropyLoss()

        self.dice_loss = DiceLoss(
            num_classes=num_classes,
        )

    def forward(self, logits, target):

        target = target.long()

        loss_ce = self.ce_loss(
            logits,
            target,
        )

        loss_dice = self.dice_loss(
            logits,
            target,
        )

        loss_seg = (
            loss_ce
            + loss_dice
        )

        return {
            "loss_seg": loss_seg,
            "loss_ce": loss_ce.detach(),
            "loss_dice": loss_dice.detach(),
        }


from typing import List, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

def sigmoid_focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
) -> torch.Tensor:

    prob = torch.sigmoid(logits)

    ce_loss = F.binary_cross_entropy_with_logits(
        logits,
        targets,
        reduction="none",
    )

    p_t = (
        prob * targets
        + (1.0 - prob) * (1.0 - targets)
    )

    focal_weight = (1.0 - p_t).pow(gamma)

    alpha_weight = (
        alpha * targets
        + (1.0 - alpha) * (1.0 - targets)
    )

    loss = (
        ce_loss
        * focal_weight
        * alpha_weight
    )

    return loss.sum()

def ltrb_to_boxes(
    points: torch.Tensor,
    ltrb: torch.Tensor,
) -> torch.Tensor:

    x = points[:, 0]
    y = points[:, 1]

    l = ltrb[:, 0]
    t = ltrb[:, 1]
    r = ltrb[:, 2]
    b = ltrb[:, 3]

    return torch.stack(
        [
            x - l,
            y - t,
            x + r,
            y + b,
        ],
        dim=1,
    )

def box_giou(
    boxes1: torch.Tensor,
    boxes2: torch.Tensor,
) -> torch.Tensor:

    x1 = torch.max(boxes1[:, 0], boxes2[:, 0])
    y1 = torch.max(boxes1[:, 1], boxes2[:, 1])
    x2 = torch.min(boxes1[:, 2], boxes2[:, 2])
    y2 = torch.min(boxes1[:, 3], boxes2[:, 3])

    inter_w = (x2 - x1).clamp(min=0)
    inter_h = (y2 - y1).clamp(min=0)

    intersection = inter_w * inter_h

    area1 = (
        (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0)
        * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0)
    )

    area2 = (
        (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0)
        * (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0)
    )

    union = area1 + area2 - intersection

    iou = intersection / union.clamp(min=1e-7)

    cx1 = torch.min(boxes1[:, 0], boxes2[:, 0])
    cy1 = torch.min(boxes1[:, 1], boxes2[:, 1])
    cx2 = torch.max(boxes1[:, 2], boxes2[:, 2])
    cy2 = torch.max(boxes1[:, 3], boxes2[:, 3])

    cover_area = (
        (cx2 - cx1).clamp(min=0)
        * (cy2 - cy1).clamp(min=0)
    )

    giou = (
        iou
        - (cover_area - union)
        / cover_area.clamp(min=1e-7)
    )

    return giou

def compute_centerness_targets(
    ltrb_targets: torch.Tensor,
) -> torch.Tensor:

    left_right = ltrb_targets[:, [0, 2]]
    top_bottom = ltrb_targets[:, [1, 3]]

    lr_min = left_right.min(dim=1).values
    lr_max = left_right.max(dim=1).values.clamp(min=1e-7)

    tb_min = top_bottom.min(dim=1).values
    tb_max = top_bottom.max(dim=1).values.clamp(min=1e-7)

    centerness = torch.sqrt(
        (lr_min / lr_max)
        * (tb_min / tb_max)
    )

    return centerness.clamp(
        min=0.0,
        max=1.0,
    )

class FCOSLoss(nn.Module):

    def __init__(
        self,
        num_classes: int,
        strides: Optional[List[int]] = None,
        center_sampling_radius: float = 3.0,
    ):
        super().__init__()

        self.num_classes = int(num_classes)

        self.strides = (
            strides
            if strides is not None
            else [4, 8, 16]
        )

        self.center_sampling_radius = float(
            center_sampling_radius
        )

        self.regress_ranges = [
            (0, 64),
            (64, 128),
            (128, 1e8),
        ]

    def forward(
        self,
        det_outputs: Dict,
        targets: List[Dict],
    ):

        cls_logits_list = det_outputs[
            "cls_logits"
        ]

        box_reg_list = det_outputs[
            "box_reg"
        ]

        centerness_list = det_outputs[
            "centerness"
        ]

        strides = det_outputs.get(
            "strides",
            self.strides,
        )

        (
            cls_flat,
            box_flat,
            ctr_flat,
            points,
            reg_ranges,
            point_strides,
        ) = self._flatten_outputs(
            cls_logits_list,
            box_reg_list,
            centerness_list,
            strides,
        )

        cls_flat = cls_flat.float()
        box_flat = box_flat.float()
        ctr_flat = ctr_flat.float()

        points = points.float()
        reg_ranges = reg_ranges.float()
        point_strides = point_strides.float()

        device = cls_flat.device

        batch_size = cls_flat.shape[0]
        num_points = cls_flat.shape[1]

        labels_targets = []
        box_targets = []

        for batch_index in range(batch_size):

            gt_boxes = targets[
                batch_index
            ]["boxes"].to(
                device
            ).float()

            gt_labels = targets[
                batch_index
            ]["labels"].to(
                device
            ).long()

            labels, boxes = self._get_targets_single_image(
                points=points,
                reg_ranges=reg_ranges,
                point_strides=point_strides,
                gt_boxes=gt_boxes,
                gt_labels=gt_labels,
            )

            labels_targets.append(labels)
            box_targets.append(boxes)

        labels_targets = torch.stack(
            labels_targets,
            dim=0,
        )

        box_targets = torch.stack(
            box_targets,
            dim=0,
        )

        pos_mask = labels_targets >= 0

        num_pos = int(
            pos_mask.sum().item()
        )

        cls_targets = torch.zeros_like(
            cls_flat
        )

        if num_pos > 0:

            pos_labels = labels_targets[
                pos_mask
            ]

            cls_targets[
                pos_mask,
                pos_labels,
            ] = 1.0

        normalizer = max(
            num_pos,
            batch_size,
        )

        loss_cls = sigmoid_focal_loss(
            logits=cls_flat.reshape(
                -1,
                self.num_classes,
            ),
            targets=cls_targets.reshape(
                -1,
                self.num_classes,
            ),
        ) / normalizer

        if num_pos > 0:

            points_batch = points.unsqueeze(
                0
            ).expand(
                batch_size,
                num_points,
                2,
            )

            pos_points = points_batch[
                pos_mask
            ]

            pos_box_pred = box_flat[
                pos_mask
            ].clamp(
                min=0.0
            )

            pos_box_target = box_targets[
                pos_mask
            ]

            pos_ctr_logits = ctr_flat[
                pos_mask
            ]

            ctr_targets = compute_centerness_targets(
                pos_box_target
            )

            pred_boxes = ltrb_to_boxes(
                pos_points,
                pos_box_pred,
            )

            target_boxes = ltrb_to_boxes(
                pos_points,
                pos_box_target,
            )

            giou = box_giou(
                pred_boxes,
                target_boxes,
            )

            loss_box = (
                (1.0 - giou)
                * ctr_targets
            ).sum() / ctr_targets.sum().clamp(
                min=1.0
            )

            loss_ctr = (
                F.binary_cross_entropy_with_logits(
                    pos_ctr_logits,
                    ctr_targets,
                    reduction="sum",
                )
                / normalizer
            )

        else:

            loss_box = cls_flat.sum() * 0.0
            loss_ctr = cls_flat.sum() * 0.0

        loss_det = (
            loss_cls
            + loss_box
            + loss_ctr
        )

        return {
            "loss_det": loss_det,
            "loss_det_cls": loss_cls.detach(),
            "loss_det_box": loss_box.detach(),
            "loss_det_ctr": loss_ctr.detach(),
            "num_pos": num_pos,
        }

    def _flatten_outputs(
        self,
        cls_logits_list,
        box_reg_list,
        centerness_list,
        strides,
    ):

        cls_all = []
        box_all = []
        ctr_all = []

        points_all = []
        reg_ranges_all = []
        point_strides_all = []

        for level, (
            cls_logits,
            box_reg,
            centerness,
            stride,
        ) in enumerate(
            zip(
                cls_logits_list,
                box_reg_list,
                centerness_list,
                strides,
            )
        ):

            batch_size, channels, height, width = (
                cls_logits.shape
            )

            cls_level = cls_logits.permute(
                0,
                2,
                3,
                1,
            ).reshape(
                batch_size,
                -1,
                channels,
            )

            box_level = box_reg.permute(
                0,
                2,
                3,
                1,
            ).reshape(
                batch_size,
                -1,
                4,
            )

            ctr_level = centerness.permute(
                0,
                2,
                3,
                1,
            ).reshape(
                batch_size,
                -1,
            )

            points = self._make_points(
                height,
                width,
                int(stride),
                cls_logits.device,
                cls_logits.dtype,
            )

            reg_min, reg_max = (
                self.regress_ranges[level]
            )

            reg_range = torch.tensor(
                [reg_min, reg_max],
                device=cls_logits.device,
                dtype=cls_logits.dtype,
            ).unsqueeze(
                0
            ).expand(
                points.shape[0],
                2,
            )

            point_stride = torch.full(
                (points.shape[0],),
                float(stride),
                device=cls_logits.device,
                dtype=cls_logits.dtype,
            )

            cls_all.append(cls_level)
            box_all.append(box_level)
            ctr_all.append(ctr_level)

            points_all.append(points)
            reg_ranges_all.append(reg_range)
            point_strides_all.append(point_stride)

        return (
            torch.cat(cls_all, dim=1),
            torch.cat(box_all, dim=1),
            torch.cat(ctr_all, dim=1),
            torch.cat(points_all, dim=0),
            torch.cat(reg_ranges_all, dim=0),
            torch.cat(point_strides_all, dim=0),
        )

    @staticmethod
    def _make_points(
        height,
        width,
        stride,
        device,
        dtype,
    ):

        shifts_x = (
            torch.arange(
                width,
                device=device,
                dtype=dtype,
            )
            + 0.5
        ) * stride

        shifts_y = (
            torch.arange(
                height,
                device=device,
                dtype=dtype,
            )
            + 0.5
        ) * stride

        yy, xx = torch.meshgrid(
            shifts_y,
            shifts_x,
            indexing="ij",
        )

        return torch.stack(
            [
                xx.reshape(-1),
                yy.reshape(-1),
            ],
            dim=1,
        )

    def _get_targets_single_image(
        self,
        points,
        reg_ranges,
        point_strides,
        gt_boxes,
        gt_labels,
    ):

        num_points = points.shape[0]

        labels = torch.full(
            (num_points,),
            -1,
            dtype=torch.long,
            device=points.device,
        )

        bbox_targets = torch.zeros(
            (num_points, 4),
            dtype=points.dtype,
            device=points.device,
        )

        if gt_boxes.numel() == 0:
            return labels, bbox_targets

        xs = points[:, 0]
        ys = points[:, 1]

        left = (
            xs[:, None]
            - gt_boxes[None, :, 0]
        )

        top = (
            ys[:, None]
            - gt_boxes[None, :, 1]
        )

        right = (
            gt_boxes[None, :, 2]
            - xs[:, None]
        )

        bottom = (
            gt_boxes[None, :, 3]
            - ys[:, None]
        )

        ltrb = torch.stack(
            [
                left,
                top,
                right,
                bottom,
            ],
            dim=2,
        )

        inside_box = (
            ltrb.min(dim=2).values
            > 0
        )

        centers_x = (
            gt_boxes[:, 0]
            + gt_boxes[:, 2]
        ) * 0.5

        centers_y = (
            gt_boxes[:, 1]
            + gt_boxes[:, 3]
        ) * 0.5

        radius = (
            point_strides[:, None]
            * self.center_sampling_radius
        )

        center_x1 = torch.maximum(
            centers_x[None, :] - radius,
            gt_boxes[None, :, 0],
        )

        center_y1 = torch.maximum(
            centers_y[None, :] - radius,
            gt_boxes[None, :, 1],
        )

        center_x2 = torch.minimum(
            centers_x[None, :] + radius,
            gt_boxes[None, :, 2],
        )

        center_y2 = torch.minimum(
            centers_y[None, :] + radius,
            gt_boxes[None, :, 3],
        )

        inside_center = (
            (xs[:, None] >= center_x1)
            & (xs[:, None] <= center_x2)
            & (ys[:, None] >= center_y1)
            & (ys[:, None] <= center_y2)
        )

        max_distance = ltrb.max(
            dim=2
        ).values

        inside_range = (
            (
                max_distance
                >= reg_ranges[:, None, 0]
            )
            &
            (
                max_distance
                <= reg_ranges[:, None, 1]
            )
        )

        candidate = (
            inside_box
            & inside_center
            & inside_range
        )

        areas = (
            (gt_boxes[:, 2] - gt_boxes[:, 0]).clamp(min=0)
            * (gt_boxes[:, 3] - gt_boxes[:, 1]).clamp(min=0)
        )

        areas = areas[
            None, :
        ].repeat(
            num_points,
            1,
        )

        areas[
            ~candidate
        ] = 1e8

        min_area, min_indices = areas.min(
            dim=1
        )

        pos_mask = (
            min_area < 1e8
        )

        if pos_mask.any():

            labels[
                pos_mask
            ] = gt_labels[
                min_indices[
                    pos_mask
                ]
            ]

            bbox_targets[
                pos_mask
            ] = ltrb[
                pos_mask,
                min_indices[
                    pos_mask
                ],
            ]

        return (
            labels,
            bbox_targets,
        )


from typing import Dict, List

import torch
import torch.nn as nn


class MultiTaskLoss(nn.Module):

    def __init__(
        self,
        seg_num_classes,
        det_num_classes,
        seg_weight=0.6,
        det_weight=0.4,
    ):
        super().__init__()

        self.seg_weight = float(seg_weight)
        self.det_weight = float(det_weight)

        self.seg_loss_fn = SegLoss(
            num_classes=seg_num_classes,
        )

        self.det_loss_fn = FCOSLoss(
            num_classes=det_num_classes,
            strides=[4, 8, 16],
        )

    def forward(
        self,
        outputs: Dict,
        batch: Dict,
    ):

        device = self._get_device(outputs)

        zero = torch.tensor(
            0.0,
            device=device,
        )

        total_loss = zero.clone()

        result = {
            "loss": zero.clone(),

            "loss_seg": zero.clone(),
            "loss_ce": zero.clone(),
            "loss_dice": zero.clone(),

            "loss_det": zero.clone(),
            "loss_det_cls": zero.clone(),
            "loss_det_box": zero.clone(),
            "loss_det_ctr": zero.clone(),

            "num_det_pos": 0,
        }

        if (
            "seg_logits" in outputs
            and outputs["seg_logits"] is not None
        ):
            seg_logits_all = outputs["seg_logits"]

            has_seg = batch.get(
                "has_seg",
                None,
            )

            if has_seg is None:
                seg_indices = list(
                    range(seg_logits_all.shape[0])
                )
            else:
                seg_indices = [
                    i
                    for i, flag in enumerate(has_seg)
                    if bool(flag)
                ]

            if len(seg_indices) > 0:

                seg_logits = seg_logits_all[
                    seg_indices
                ]

                masks = batch["masks"][
                    seg_indices
                ].to(
                    device
                ).long()

                seg_loss_dict = self.seg_loss_fn(
                    seg_logits,
                    masks,
                )

                loss_seg = seg_loss_dict[
                    "loss_seg"
                ]

                total_loss = (
                    total_loss
                    + self.seg_weight * loss_seg
                )

                result["loss_seg"] = (
                    loss_seg.detach()
                )

                result["loss_ce"] = (
                    seg_loss_dict["loss_ce"]
                )

                result["loss_dice"] = (
                    seg_loss_dict["loss_dice"]
                )

        if (
            "det_outputs" in outputs
            and outputs["det_outputs"] is not None
        ):

            det_outputs_all = outputs[
                "det_outputs"
            ]

            batch_size = len(
                batch["boxes"]
            )

            has_det = batch.get(
                "has_det",
                None,
            )

            if has_det is None:
                det_indices = list(
                    range(batch_size)
                )
            else:
                det_indices = [
                    i
                    for i, flag in enumerate(has_det)
                    if bool(flag)
                ]

            if len(det_indices) > 0:

                det_outputs = (
                    self._select_det_outputs(
                        det_outputs_all,
                        det_indices,
                    )
                )

                targets = []

                for i in det_indices:

                    boxes = batch["boxes"][
                        i
                    ].to(
                        device
                    ).float()

                    labels = batch["labels"][
                        i
                    ].to(
                        device
                    ).long()

                    if boxes.numel() == 0:
                        boxes = boxes.reshape(
                            0,
                            4,
                        )

                        labels = labels.reshape(
                            0
                        )

                    targets.append(
                        {
                            "boxes": boxes,
                            "labels": labels,
                        }
                    )

                det_loss_dict = (
                    self.det_loss_fn(
                        det_outputs,
                        targets,
                    )
                )

                loss_det = det_loss_dict[
                    "loss_det"
                ]

                total_loss = (
                    total_loss
                    + self.det_weight * loss_det
                )

                result["loss_det"] = (
                    loss_det.detach()
                )

                result["loss_det_cls"] = (
                    det_loss_dict[
                        "loss_det_cls"
                    ]
                )

                result["loss_det_box"] = (
                    det_loss_dict[
                        "loss_det_box"
                    ]
                )

                result["loss_det_ctr"] = (
                    det_loss_dict[
                        "loss_det_ctr"
                    ]
                )

                result["num_det_pos"] = (
                    det_loss_dict[
                        "num_pos"
                    ]
                )

        result["loss"] = total_loss

        return result

    @staticmethod
    def _select_det_outputs(
        det_outputs: Dict,
        indices: List[int],
    ):

        selected = {}

        for key, value in det_outputs.items():

            if key in [
                "cls_logits",
                "box_reg",
                "centerness",
            ]:
                selected[key] = [
                    feature[indices]
                    for feature in value
                ]

            else:

                selected[key] = value

        return selected

    @staticmethod
    def _get_device(outputs: Dict):

        if (
            "seg_logits" in outputs
            and outputs["seg_logits"] is not None
        ):
            return outputs[
                "seg_logits"
            ].device

        if (
            "det_outputs" in outputs
            and outputs["det_outputs"] is not None
        ):
            return outputs[
                "det_outputs"
            ]["cls_logits"][0].device

        return torch.device("cpu")

def build_loss_from_cfg(cfg: Dict):

    task_info = get_task_info(cfg)

    loss_cfg = cfg.get(
        "loss",
        {},
    )

    seg_weight = float(
        loss_cfg.get(
            "seg_weight",
            0.6,
        )
    )

    det_weight = float(
        loss_cfg.get(
            "det_weight",
            0.4,
        )
    )

    criterion = MultiTaskLoss(
        seg_num_classes=task_info.seg_num_classes,
        det_num_classes=task_info.det_num_classes,
        seg_weight=seg_weight,
        det_weight=det_weight,
    )

    return criterion
