from dataclasses import dataclass
from typing import Dict, List, Any

@dataclass
class TaskInfo:

    seg_class_names: List[str]
    det_class_names: List[str]

    seg_num_classes: int
    det_num_classes: int

    seg_name_to_id: Dict[str, int]
    det_name_to_id: Dict[str, int]

    seg_bg_index: int = 0

def _normalize_names(names: Any, key: str) -> List[str]:

    if names is None:
        raise KeyError(f"config.task.{key} 缺失。")

    if isinstance(names, list):
        out = [str(x) for x in names]

    elif isinstance(names, dict):

        items = sorted(names.items(), key=lambda x: int(x[0]))
        out = [str(v) for _, v in items]

    else:
        raise TypeError(
            f"config.task.{key} 必须是 list 或 dict，当前类型是 {type(names)}"
        )

    if len(out) == 0:
        raise ValueError(f"config.task.{key} 不能为空。")

    if len(set(out)) != len(out):
        raise ValueError(f"config.task.{key} 中存在重复类别名: {out}")

    return out

def get_task_info(cfg: Dict) -> TaskInfo:

    task_cfg = cfg.get("task", {})

    seg_class_names = _normalize_names(
        task_cfg.get("seg_class_names", None),
        "seg_class_names",
    )

    det_class_names = _normalize_names(
        task_cfg.get("det_class_names", None),
        "det_class_names",
    )

    if seg_class_names[0] != "BG":
        raise ValueError(
            f"分割第 0 类必须是 BG，当前 seg_class_names[0]={seg_class_names[0]}"
        )

    if "BG" in det_class_names:
        raise ValueError(
            f"检测类别不应该包含 BG，当前 det_class_names={det_class_names}"
        )

    seg_name_to_id = {name: i for i, name in enumerate(seg_class_names)}
    det_name_to_id = {name: i for i, name in enumerate(det_class_names)}

    return TaskInfo(
        seg_class_names=seg_class_names,
        det_class_names=det_class_names,
        seg_num_classes=len(seg_class_names),
        det_num_classes=len(det_class_names),
        seg_name_to_id=seg_name_to_id,
        det_name_to_id=det_name_to_id,
        seg_bg_index=0,
    )

def print_task_info(task_info: TaskInfo):

    print("=" * 100)
    print("[Task Info]")
    print(f"seg_num_classes: {task_info.seg_num_classes}")
    print(f"seg_class_names: {task_info.seg_class_names}")
    print(f"det_num_classes: {task_info.det_num_classes}")
    print(f"det_class_names: {task_info.det_class_names}")
    print("=" * 100)


import torch
import torch.nn as nn

class DSC(nn.Module):

    def __init__(self, c_in, c_out, k_size=3, stride=1, padding=1):

        super().__init__()

        self.c_in = c_in
        self.c_out = c_out

        self.dw = nn.Conv2d(
            c_in,
            c_in,
            kernel_size=k_size,
            stride=stride,
            padding=padding,
            groups=c_in,
        )

        self.pw = nn.Conv2d(
            c_in,
            c_out,
            kernel_size=1,
            stride=1,
            padding=0,
        )

    def forward(self, x):

        x = self.dw(x)
        x = self.pw(x)
        return x

class IDSC(nn.Module):

    def __init__(self, c_in, c_out, k_size=3, stride=1, padding=1):

        super().__init__()

        self.c_in = c_in
        self.c_out = c_out

        self.pw = nn.Conv2d(
            c_in,
            c_out,
            kernel_size=1,
            stride=1,
            padding=0,
        )

        self.dw = nn.Conv2d(
            c_out,
            c_out,
            kernel_size=k_size,
            stride=stride,
            padding=padding,
            groups=c_out,
        )

    def forward(self, x):

        x = self.pw(x)
        x = self.dw(x)
        return x


import torch.nn as nn

class AddFusion(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, cnn_feat, transformer_feat):

        return cnn_feat + transformer_feat


import torch
import torch.nn as nn


class CNN(nn.Module):

    def __init__(self):

        super().__init__()

        self.conv1_1 = DSC(3, 32)
        self.conv1_2 = DSC(32, 32)
        self.conv1_3 = DSC(32, 32)

        self.c1_1 = nn.Sequential(self.conv1_1, nn.GroupNorm(16, 32), nn.GELU())
        self.c1_2 = nn.Sequential(self.conv1_2, nn.GroupNorm(16, 32), nn.GELU())
        self.c1_3 = nn.Sequential(self.conv1_3, nn.GroupNorm(16, 32), nn.GELU())

        self.res1 = nn.Conv2d(64, 32, kernel_size=1)
        self.pool1 = DSC(32, 32, 2, 2, 0)

        self.conv2_1 = DSC(32, 64)
        self.conv2_2 = DSC(64, 64)
        self.conv2_3 = DSC(64, 64)

        self.c2_1 = nn.Sequential(self.conv2_1, nn.GroupNorm(16, 64), nn.GELU())
        self.c2_2 = nn.Sequential(self.conv2_2, nn.GroupNorm(16, 64), nn.GELU())
        self.c2_3 = nn.Sequential(self.conv2_3, nn.GroupNorm(16, 64), nn.GELU())

        self.res2 = nn.Conv2d(128, 64, kernel_size=1)
        self.pool2 = DSC(64, 64, 2, 2, 0)

        self.conv3_1 = DSC(64, 128)
        self.conv3_2 = DSC(128, 128)
        self.conv3_3 = DSC(128, 128)
        self.conv3_4 = DSC(128, 128)
        self.conv3_5 = DSC(128, 128)

        self.c3_1 = nn.Sequential(self.conv3_1, nn.GroupNorm(16, 128), nn.GELU())
        self.c3_2 = nn.Sequential(self.conv3_2, nn.GroupNorm(16, 128), nn.GELU())
        self.c3_3 = nn.Sequential(self.conv3_3, nn.GroupNorm(16, 128), nn.GELU())
        self.c3_4 = nn.Sequential(self.conv3_4, nn.GroupNorm(16, 128), nn.GELU())
        self.c3_5 = nn.Sequential(self.conv3_5, nn.GroupNorm(16, 128), nn.GELU())

        self.res3 = nn.Conv2d(256, 128, kernel_size=1)
        self.pool3 = DSC(128, 128, 2, 2, 0)

        self.conv4_1 = DSC(128, 256)
        self.conv4_2 = DSC(256, 256)
        self.conv4_3 = DSC(256, 256)

        self.c4_1 = nn.Sequential(self.conv4_1, nn.GroupNorm(16, 256), nn.GELU())
        self.c4_2 = nn.Sequential(self.conv4_2, nn.GroupNorm(16, 256), nn.GELU())
        self.c4_3 = nn.Sequential(self.conv4_3, nn.GroupNorm(16, 256), nn.GELU())

        self.res4 = nn.Conv2d(512, 256, kernel_size=1)
        self.pool4 = DSC(256, 256, 2, 2, 0)

        self.pool5 = DSC(256, 512, 2, 2, 0)

    def forward(self, x):

        x1_1 = self.c1_1(x)
        x1_p = self.pool1(x1_1)
        x1_2 = self.c1_2(x1_p)
        x1_3 = self.c1_3(x1_2)
        temp1 = torch.cat([x1_p, x1_3], dim=1)
        x1_out = self.res1(temp1)

        x2_1 = self.c2_1(x1_out)
        x2_p = self.pool2(x2_1)
        x2_2 = self.c2_2(x2_p)
        x2_3 = self.c2_3(x2_2)
        temp2 = torch.cat([x2_p, x2_3], dim=1)
        x2_out = self.res2(temp2)

        x3_1 = self.c3_1(x2_out)
        x3_2 = self.c3_2(x3_1)
        x3_p = self.pool3(x3_2)
        x3_3 = self.c3_3(x3_p)
        x3_4 = self.c3_4(x3_3)
        x3_5 = self.c3_5(x3_4)
        temp3 = torch.cat([x3_p, x3_5], dim=1)
        x3_out = self.res3(temp3)

        x4_1 = self.c4_1(x3_out)
        x4_p = self.pool4(x4_1)
        x4_2 = self.c4_2(x4_p)
        x4_3 = self.c4_3(x4_2)
        temp4 = torch.cat([x4_p, x4_3], dim=1)
        x4_out = self.res4(temp4)

        out = self.pool5(x4_out)

        return x1_out, x2_out, x3_out, x4_out, out


import torch
import torch.nn as nn
import timm

class ConvGNAct(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                bias=False,
            ),
            nn.GroupNorm(16, out_channels),
            nn.GELU(),
        )

    def forward(self, x):
        return self.block(x)

class ProjectionHead(nn.Module):

    def __init__(self, in_channels):
        super().__init__()

        self.proj1 = ConvGNAct(
            in_channels[0],
            64,
        )

        self.proj2 = ConvGNAct(
            in_channels[1],
            128,
        )

        self.proj3 = ConvGNAct(
            in_channels[2],
            256,
        )

        self.proj4 = ConvGNAct(
            in_channels[3],
            512,
        )

    def forward(self, y1, y2, y3, out):

        y1 = self.proj1(y1)
        y2 = self.proj2(y2)
        y3 = self.proj3(y3)
        out = self.proj4(out)

        return y1, y2, y3, out

class Transformer(nn.Module):

    def __init__(self):
        super().__init__()

        self.backbone = timm.create_model(
            "swin_tiny_patch4_window7_224",
            features_only=True,
            out_indices=(0, 1, 2, 3),
            strict_img_size=False,
        )

        in_channels = self.backbone.feature_info.channels()

        self.projection = ProjectionHead(
            in_channels
        )

    @staticmethod
    def to_nchw(x):

        return x.permute(
            0,
            3,
            1,
            2,
        ).contiguous()

    def forward(self, x):

        y1, y2, y3, out = self.backbone(x)

        y1 = self.to_nchw(y1)
        y2 = self.to_nchw(y2)
        y3 = self.to_nchw(y3)
        out = self.to_nchw(out)

        y1, y2, y3, out = self.projection(
            y1,
            y2,
            y3,
            out,
        )

        return y1, y2, y3, out


import torch.nn as nn
import torch.nn.functional as F

def conv_gn_act(
    in_channels,
    out_channels,
):

    return nn.Sequential(
        nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            bias=False,
        ),
        nn.GroupNorm(16, out_channels),
        nn.GELU(),
    )

class Neck(nn.Module):

    def __init__(self):
        super().__init__()

        self.top = conv_gn_act(512, 256)

        self.lat3 = conv_gn_act(256, 256)

        self.reduce_p5_to_p4 = conv_gn_act(256, 128)

        self.lat2 = conv_gn_act(128, 128)

        self.reduce_p4_to_p3 = conv_gn_act(128, 64)

        self.lat1 = conv_gn_act(64, 64)

    def forward(
        self,
        shared_out,
        shared_f1,
        shared_f2,
        shared_f3,
    ):

        top = self.top(shared_out)

        top = F.interpolate(
            top,
            size=shared_f3.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        p5 = self.lat3(shared_f3) + top

        p5_up = self.reduce_p5_to_p4(p5)

        p5_up = F.interpolate(
            p5_up,
            size=shared_f2.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        p4 = self.lat2(shared_f2) + p5_up

        p4_up = self.reduce_p4_to_p3(p4)

        p4_up = F.interpolate(
            p4_up,
            size=shared_f1.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        p3 = self.lat1(shared_f1) + p4_up

        return [
            p3,
            p4,
            p5,
        ]


import torch
import torch.nn as nn
import torch.nn.functional as F


def ConvBlock(
    in_channels,
    out_channels,
    kernel_size=1,
    padding=0,
):

    return nn.Sequential(
        nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=False,
        ),
        nn.GroupNorm(16, out_channels),
        nn.GELU(),
    )

def DscBlock(
    in_channels,
    out_channels,
):

    return nn.Sequential(
        DSC(
            in_channels,
            out_channels,
            k_size=3,
            stride=1,
            padding=1,
        ),
        nn.GroupNorm(16, out_channels),
        nn.GELU(),

        DSC(
            out_channels,
            out_channels,
            k_size=3,
            stride=1,
            padding=1,
        ),
        nn.GroupNorm(16, out_channels),
        nn.GELU(),
    )

class Decoder(nn.Module):

    def __init__(self, num_classes):
        super().__init__()

        self.num_classes = int(num_classes)

        self.top = DscBlock(512, 256)

        self.top_lateral = ConvBlock(256, 128)

        self.lat3 = ConvBlock(256, 128)

        self.lat2 = ConvBlock(128, 128)

        self.lat1 = ConvBlock(64, 128)

        self.aggregate = ConvBlock(512, 96)

        self.pred = nn.Conv2d(
            96,
            self.num_classes,
            kernel_size=1,
        )

    def forward(
        self,
        shared_out,
        shared_f1,
        shared_f2,
        shared_f3,
    ):

        top = self.top(shared_out)

        p_top = self.top_lateral(top)

        p3 = self.lat3(shared_f3)

        p3 = p3 + F.interpolate(
            p_top,
            size=shared_f3.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        p2 = self.lat2(shared_f2)

        p2 = p2 + F.interpolate(
            p3,
            size=shared_f2.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        p1 = self.lat1(shared_f1)

        p1 = p1 + F.interpolate(
            p2,
            size=shared_f1.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        output_size = p1.shape[-2:]

        multi_scale = torch.cat(
            [
                p1,

                F.interpolate(
                    p2,
                    size=output_size,
                    mode="bilinear",
                    align_corners=False,
                ),

                F.interpolate(
                    p3,
                    size=output_size,
                    mode="bilinear",
                    align_corners=False,
                ),

                F.interpolate(
                    p_top,
                    size=output_size,
                    mode="bilinear",
                    align_corners=False,
                ),
            ],
            dim=1,
        )

        x = self.aggregate(multi_scale)

        seg_logits = self.pred(x)

        seg_logits = F.interpolate(
            seg_logits,
            scale_factor=4,
            mode="bilinear",
            align_corners=False,
        )

        return seg_logits


import math
from typing import Dict, Tuple

import torch
import torch.nn as nn
from torchvision.ops import batched_nms

class ConvGNAct(nn.Module):

    def __init__(self, in_ch, out_ch, k=3, s=1, p=1, gn_groups=16):
        super().__init__()

        g = max(1, min(gn_groups, out_ch // 8))
        while out_ch % g != 0 and g > 1:
            g -= 1

        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=False)
        self.gn = nn.GroupNorm(g, out_ch)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(self.gn(self.conv(x)))

class Scale(nn.Module):

    def __init__(self, init_value=1.0):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(float(init_value)))

    def forward(self, x):
        return x * self.scale

class DetHead(nn.Module):

    def __init__(
        self,
        in_channels,
        num_classes,
        feat_ch=128,
        gn_groups=16,
        level_strides=None,
    ):
        super().__init__()

        self.num_classes = int(num_classes)
        self.num_levels = len(in_channels)

        if level_strides is None:
            self.level_strides = [4 * (2 ** i) for i in range(self.num_levels)]
        else:
            self.level_strides = [int(s) for s in level_strides]

        if len(self.level_strides) != self.num_levels:
            raise ValueError(
                f"level_strides 数量必须与检测层数量一致："
                f"{len(self.level_strides)} vs {self.num_levels}"
            )

        self.stem = nn.ModuleList([
            nn.Conv2d(channels, feat_ch, kernel_size=1)
            for channels in in_channels
        ])

        self.cls_tower = nn.Sequential(
            ConvGNAct(feat_ch, feat_ch, gn_groups=gn_groups),
            ConvGNAct(feat_ch, feat_ch, gn_groups=gn_groups),
            ConvGNAct(feat_ch, feat_ch, gn_groups=gn_groups),
            ConvGNAct(feat_ch, feat_ch, gn_groups=gn_groups),
        )

        self.box_tower = nn.Sequential(
            ConvGNAct(feat_ch, feat_ch, gn_groups=gn_groups),
            ConvGNAct(feat_ch, feat_ch, gn_groups=gn_groups),
            ConvGNAct(feat_ch, feat_ch, gn_groups=gn_groups),
            ConvGNAct(feat_ch, feat_ch, gn_groups=gn_groups),
        )

        self.cls_pred = nn.Conv2d(feat_ch, self.num_classes, kernel_size=3, padding=1)
        self.box_pred = nn.Conv2d(feat_ch, 4, kernel_size=3, padding=1)
        self.ctr_pred = nn.Conv2d(feat_ch, 1, kernel_size=3, padding=1)

        self.scales = nn.ModuleList([Scale(1.0) for _ in range(self.num_levels)])

        self._init_weights()

    def _init_weights(self):

        for layer in [self.cls_pred, self.box_pred, self.ctr_pred]:
            nn.init.normal_(layer.weight, std=0.01)
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0.0)

        prior_prob = 0.01
        bias = math.log(prior_prob / (1.0 - prior_prob))

        with torch.no_grad():
            self.cls_pred.bias.fill_(bias)

    def forward(self, feats):

        if len(feats) != self.num_levels:
            raise ValueError(
                f"DetHead 接收到 {len(feats)} 个特征层，"
                f"但初始化时要求 {self.num_levels} 个。"
            )

        cls_logits_list = []
        box_reg_list = []
        centerness_list = []

        for level, feature in enumerate(feats):

            feature = self.stem[level](feature)

            cls_feature = self.cls_tower(feature)
            box_feature = self.box_tower(feature)

            cls_logits = self.cls_pred(cls_feature)

            bbox_raw = self.box_pred(box_feature)

            bbox_ltrb = torch.exp(
                self.scales[level](bbox_raw).clamp(min=-5, max=5)
            ) * float(self.level_strides[level])

            centerness = self.ctr_pred(box_feature)

            cls_logits_list.append(cls_logits)
            box_reg_list.append(bbox_ltrb)
            centerness_list.append(centerness)

        return {
            "cls_logits": cls_logits_list,
            "box_reg": box_reg_list,
            "centerness": centerness_list,
            "strides": self.level_strides,
        }

    @torch.no_grad()
    def decode_predictions(
        self,
        det_outputs: Dict,
        img_size: Tuple[int, int],
        conf_thres: float = 0.25,
        nms_iou: float = 0.50,
        max_detections: int = 300,
    ):

        cls_logits_list = det_outputs["cls_logits"]
        box_reg_list = det_outputs["box_reg"]
        centerness_list = det_outputs["centerness"]
        strides = det_outputs.get("strides", self.level_strides)

        batch_size = cls_logits_list[0].shape[0]
        device = cls_logits_list[0].device

        img_h, img_w = int(img_size[0]), int(img_size[1])

        all_boxes = [[] for _ in range(batch_size)]
        all_scores = [[] for _ in range(batch_size)]
        all_labels = [[] for _ in range(batch_size)]

        for cls_logits, box_reg, centerness, stride in zip(
            cls_logits_list,
            box_reg_list,
            centerness_list,
            strides,
        ):
            batch, channels, height, width = cls_logits.shape

            points = self._make_points(
                height=height,
                width=width,
                stride=int(stride),
                device=device,
                dtype=cls_logits.dtype,
            )

            cls_prob = torch.sigmoid(cls_logits)
            ctr_prob = torch.sigmoid(centerness)

            cls_prob = cls_prob.permute(0, 2, 3, 1).reshape(batch, -1, channels)

            ctr_prob = ctr_prob.permute(0, 2, 3, 1).reshape(batch, -1, 1)

            scores = torch.sqrt(cls_prob * ctr_prob)

            box_reg_flat = box_reg.permute(0, 2, 3, 1).reshape(batch, -1, 4)

            boxes = self._ltrb_to_boxes(points, box_reg_flat)

            boxes[..., 0::2] = boxes[..., 0::2].clamp(0, img_w - 1)
            boxes[..., 1::2] = boxes[..., 1::2].clamp(0, img_h - 1)

            for batch_idx in range(batch_size):
                scores_i = scores[batch_idx]
                boxes_i = boxes[batch_idx]

                keep = scores_i > float(conf_thres)

                if not keep.any():
                    continue

                point_indices, labels = torch.where(keep)

                scores_keep = scores_i[point_indices, labels]
                boxes_keep = boxes_i[point_indices]

                all_boxes[batch_idx].append(boxes_keep)
                all_scores[batch_idx].append(scores_keep)
                all_labels[batch_idx].append(labels.long())

        results = []

        for batch_idx in range(batch_size):

            if len(all_boxes[batch_idx]) == 0:
                results.append({
                    "boxes": torch.zeros((0, 4), dtype=torch.float32, device=device),
                    "scores": torch.zeros((0,), dtype=torch.float32, device=device),
                    "labels": torch.zeros((0,), dtype=torch.long, device=device),
                })
                continue

            boxes = torch.cat(all_boxes[batch_idx], dim=0)
            scores = torch.cat(all_scores[batch_idx], dim=0)
            labels = torch.cat(all_labels[batch_idx], dim=0)

            valid = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
            boxes = boxes[valid]
            scores = scores[valid]
            labels = labels[valid]

            if boxes.numel() == 0:
                results.append({
                    "boxes": torch.zeros((0, 4), dtype=torch.float32, device=device),
                    "scores": torch.zeros((0,), dtype=torch.float32, device=device),
                    "labels": torch.zeros((0,), dtype=torch.long, device=device),
                })
                continue

            keep = batched_nms(boxes, scores, labels, float(nms_iou))

            keep = keep[:int(max_detections)]

            results.append({
                "boxes": boxes[keep],
                "scores": scores[keep],
                "labels": labels[keep],
            })

        return results

    @staticmethod
    def _make_points(height, width, stride, device, dtype):

        shifts_x = (torch.arange(width, device=device, dtype=dtype) + 0.5) * stride
        shifts_y = (torch.arange(height, device=device, dtype=dtype) + 0.5) * stride

        yy, xx = torch.meshgrid(shifts_y, shifts_x, indexing="ij")

        return torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=1)

    @staticmethod
    def _ltrb_to_boxes(points, ltrb):

        x = points[:, 0]
        y = points[:, 1]

        l = ltrb[..., 0]
        t = ltrb[..., 1]
        r = ltrb[..., 2]
        b = ltrb[..., 3]

        x1 = x.unsqueeze(0) - l
        y1 = y.unsqueeze(0) - t
        x2 = x.unsqueeze(0) + r
        y2 = y.unsqueeze(0) + b

        return torch.stack([x1, y1, x2, y2], dim=-1)


import torch
import torch.nn as nn

class BridgeDualMTL(nn.Module):

    def __init__(
        self,
        seg_num_classes,
        det_num_classes,
    ):
        super().__init__()

        self.seg_num_classes = int(seg_num_classes)

        self.det_num_classes = int(det_num_classes)

        self.encoder1 = CNN()

        self.encoder2 = Transformer()

        self.fuse1 = AddFusion()

        self.fuse2 = AddFusion()

        self.fuse3 = AddFusion()

        self.Conv = nn.Sequential(
            IDSC(1024, 512),
            nn.GroupNorm(16, 512),
            nn.GELU(),
        )

        self.avg = nn.AdaptiveAvgPool2d((1, 1))

        self.linear_x = nn.Linear(512, 512)

        self.linear_y = nn.Linear(512, 512)

        self.seg_decoder = Decoder(num_classes=self.seg_num_classes,)

        self.det_neck = Neck()

        self.det_head = DetHead(
            in_channels=[64, 128, 256],
            num_classes=self.det_num_classes,
            feat_ch=128,
            gn_groups=16,
            level_strides=[4, 8, 16],
        )

    def forward(self, x):

        x1, x2, x3, x4, out1 = self.encoder1(x)

        y1, y2, y3, out2 = self.encoder2(x)

        shared_f1 = self.fuse1(x2, y1)

        shared_f2 = self.fuse2(x3, y2)

        shared_f3 = self.fuse3(x4, y3)

        batch_size = out1.shape[0]
        channels = out1.shape[1]

        x_weight = torch.sigmoid(
            self.linear_x(
                self.avg(out1).reshape(
                    batch_size,
                    1,
                    1,
                    channels,
                )
            )
        )

        y_weight = torch.sigmoid(
            self.linear_y(
                self.avg(out2).reshape(
                    batch_size,
                    1,
                    1,
                    channels,
                )
            )
        )

        out1 = out1.permute(0, 2, 3, 1) * x_weight

        out2 = out2.permute(0, 2, 3, 1) * y_weight

        top_feature = torch.cat(
            [
                out1.permute(0, 3, 1, 2),
                out2.permute(0, 3, 1, 2),
            ],
            dim=1,
        )

        shared_out = self.Conv(top_feature)

        seg_logits = self.seg_decoder(
            shared_out,
            shared_f1,
            shared_f2,
            shared_f3,
        )

        det_feats = self.det_neck(
            shared_out,
            shared_f1,
            shared_f2,
            shared_f3,
        )

        det_outputs = self.det_head(det_feats)

        return {
            "seg_logits": seg_logits,
            "det_outputs": det_outputs,
        }


def build_model_from_cfg(cfg):

    task_info = get_task_info(cfg)

    model = BridgeDualMTL(
        seg_num_classes=task_info.seg_num_classes,
        det_num_classes=task_info.det_num_classes,
    )

    return model
