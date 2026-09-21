from typing import Any, Dict, List, Optional

import numpy as np
import torch

def _normalize_class_names(class_names: Optional[Any], num_classes: int) -> List[str]:

    if class_names is None:
        return [str(i) for i in range(num_classes)]

    if isinstance(class_names, list):
        names = [str(x) for x in class_names]
    elif isinstance(class_names, dict):
        items = sorted(class_names.items(), key=lambda x: int(x[0]))
        names = [str(value) for _, value in items]
    else:
        raise TypeError(f"class_names 必须为 list / dict / None，当前类型={type(class_names)}")

    if len(names) != num_classes:
        raise ValueError(f"类别名称数量 {len(names)} 与 num_classes={num_classes} 不一致。")

    if len(set(names)) != len(names):
        raise ValueError(f"class_names 中存在重复类别：{names}")

    return names

class SegmentationMetric:

    def __init__(
        self,
        num_classes: int,
        class_names: Optional[Any] = None,
        ignore_index: Optional[int] = None,
        include_bg: bool = True,
    ):
        self.num_classes = int(num_classes)

        if self.num_classes <= 1:
            raise ValueError(f"num_classes 必须大于 1，当前={self.num_classes}")

        self.class_names = _normalize_class_names(class_names, self.num_classes)
        self.ignore_index = ignore_index
        self.include_bg = bool(include_bg)

        self.hist = np.zeros((self.num_classes, self.num_classes), dtype=np.float64)

    def reset(self):

        self.hist[...] = 0.0

    @torch.no_grad()
    def update(self, pred, target):

        if torch.is_tensor(pred):
            pred = pred.detach()

            if pred.ndim == 4:
                pred = torch.argmax(pred, dim=1)

            pred = pred.cpu().numpy()

        else:
            pred = np.asarray(pred)

            if pred.ndim == 4:
                pred = np.argmax(pred, axis=1)

        if torch.is_tensor(target):
            target = target.detach().cpu().numpy()
        else:
            target = np.asarray(target)

        if pred.shape != target.shape:
            raise ValueError(f"pred 与 target 尺寸不一致：pred={pred.shape}, target={target.shape}")

        pred = pred.astype(np.int64).reshape(-1)
        target = target.astype(np.int64).reshape(-1)

        valid = (target >= 0) & (target < self.num_classes)

        if self.ignore_index is not None:
            valid &= target != int(self.ignore_index)

        pred = pred[valid]
        target = target[valid]

        if pred.size == 0:
            return

        valid_pred = (pred >= 0) & (pred < self.num_classes)
        pred = pred[valid_pred]
        target = target[valid_pred]

        if pred.size == 0:
            return

        hist = np.bincount(
            self.num_classes * target + pred,
            minlength=self.num_classes ** 2,
        ).reshape(self.num_classes, self.num_classes)

        self.hist += hist

    def compute(self) -> Dict:

        hist = self.hist.copy()

        tp = np.diag(hist)

        gt = hist.sum(axis=1)

        pred = hist.sum(axis=0)

        union = gt + pred - tp

        iou = np.divide(
            tp,
            union,
            out=np.zeros_like(tp, dtype=np.float64),
            where=union > 0,
        )

        dice = np.divide(
            2.0 * tp,
            gt + pred,
            out=np.zeros_like(tp, dtype=np.float64),
            where=(gt + pred) > 0,
        )

        precision = np.divide(
            tp,
            pred,
            out=np.zeros_like(tp, dtype=np.float64),
            where=pred > 0,
        )

        recall = np.divide(
            tp,
            gt,
            out=np.zeros_like(tp, dtype=np.float64),
            where=gt > 0,
        )

        total_pixels = hist.sum()
        pixel_acc = float(tp.sum() / max(total_pixels, 1.0))

        if self.include_bg:
            mean_indices = np.arange(self.num_classes)
        else:
            mean_indices = np.arange(1, self.num_classes)

        if self.ignore_index is not None:
            mean_indices = np.asarray(
                [i for i in mean_indices if i != int(self.ignore_index)],
                dtype=np.int64,
            )

        if len(mean_indices) == 0:
            miou = mdice = mprecision = mrecall = 0.0
        else:
            miou = float(np.mean(iou[mean_indices]))
            mdice = float(np.mean(dice[mean_indices]))
            mprecision = float(np.mean(precision[mean_indices]))
            mrecall = float(np.mean(recall[mean_indices]))

        per_class = {}

        for class_id, class_name in enumerate(self.class_names):
            per_class[class_id] = {
                "name": class_name,
                "IoU": float(iou[class_id]),
                "Dice": float(dice[class_id]),
                "Precision": float(precision[class_id]),
                "Recall": float(recall[class_id]),
                "GT_pixels": float(gt[class_id]),
                "Pred_pixels": float(pred[class_id]),
                "TP_pixels": float(tp[class_id]),
            }

        return {
            "mIoU": miou,
            "mDice": mdice,
            "PixelAcc": pixel_acc,
            "mPrecision": mprecision,
            "mRecall": mrecall,

            "IoU_per_class": iou.tolist(),
            "Dice_per_class": dice.tolist(),
            "Precision_per_class": precision.tolist(),
            "Recall_per_class": recall.tolist(),

            "GT_pixels_per_class": gt.tolist(),
            "Pred_pixels_per_class": pred.tolist(),
            "TP_pixels_per_class": tp.tolist(),

            "per_class": per_class,
            "hist": hist,
        }

    def summary_string(self) -> str:

        result = self.compute()

        return (
            f"mIoU={result['mIoU']:.4f} | "
            f"mDice={result['mDice']:.4f} | "
            f"PixelAcc={result['PixelAcc']:.4f}"
        )

    def class_summary_string(self) -> str:

        result = self.compute()
        parts = []

        for class_id in range(self.num_classes):
            r = result["per_class"][class_id]
            parts.append(f"{r['name']}: IoU={r['IoU']:.3f}, Dice={r['Dice']:.3f}")

        return " | ".join(parts)

    def detailed_class_summary_string(self) -> str:

        result = self.compute()
        parts = []

        for class_id in range(self.num_classes):
            r = result["per_class"][class_id]
            parts.append(
                f"{r['name']}: IoU={r['IoU']:.3f}, Dice={r['Dice']:.3f}, "
                f"P={r['Precision']:.3f}, R={r['Recall']:.3f}"
            )

        return " | ".join(parts)


from typing import Any, Dict, List, Optional

import numpy as np
import torch

def _normalize_class_names(class_names: Optional[Any], num_classes: int) -> List[str]:

    if class_names is None:
        return [str(i) for i in range(num_classes)]

    if isinstance(class_names, list):
        names = [str(x) for x in class_names]
    elif isinstance(class_names, dict):
        items = sorted(class_names.items(), key=lambda x: int(x[0]))
        names = [str(value) for _, value in items]
    else:
        raise TypeError(f"class_names 必须为 list / dict / None，当前类型={type(class_names)}")

    if len(names) != num_classes:
        raise ValueError(f"类别名称数量 {len(names)} 与 num_classes={num_classes} 不一致。")

    if len(set(names)) != len(names):
        raise ValueError(f"class_names 中存在重复类别：{names}")

    return names

def to_numpy(x):

    if x is None:
        return None

    if torch.is_tensor(x):
        return x.detach().cpu().numpy()

    return np.asarray(x)

def box_iou(boxes1, boxes2):

    boxes1 = np.asarray(boxes1, dtype=np.float32).reshape(-1, 4)
    boxes2 = np.asarray(boxes2, dtype=np.float32).reshape(-1, 4)

    if boxes1.shape[0] == 0 or boxes2.shape[0] == 0:
        return np.zeros((boxes1.shape[0], boxes2.shape[0]), dtype=np.float32)

    left_top = np.maximum(boxes1[:, None, :2], boxes2[None, :, :2])
    right_bottom = np.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh = np.clip(right_bottom - left_top, a_min=0.0, a_max=None)
    intersection = wh[:, :, 0] * wh[:, :, 1]

    area1 = np.clip(boxes1[:, 2] - boxes1[:, 0], 0.0, None) * np.clip(boxes1[:, 3] - boxes1[:, 1], 0.0, None)
    area2 = np.clip(boxes2[:, 2] - boxes2[:, 0], 0.0, None) * np.clip(boxes2[:, 3] - boxes2[:, 1], 0.0, None)

    union = area1[:, None] + area2[None, :] - intersection
    iou = intersection / np.clip(union, a_min=1e-12, a_max=None)

    return iou.astype(np.float32)

def compute_ap(recall, precision):

    recall = np.asarray(recall, dtype=np.float32)
    precision = np.asarray(precision, dtype=np.float32)

    if recall.size == 0 or precision.size == 0:
        return 0.0

    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))

    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])

    x = np.linspace(0.0, 1.0, 101)
    ap = np.interp(x, mrec, mpre).mean()

    return float(ap)

class DetectionMetric:

    def __init__(
        self,
        num_classes: int,
        class_names: Optional[Any] = None,
        conf_thres: float = 0.25,
        pr_iou_thres: float = 0.50,
        map_iou_start: float = 0.50,
        map_iou_end: float = 0.95,
        map_iou_step: float = 0.05,
    ):
        self.num_classes = int(num_classes)

        if self.num_classes <= 0:
            raise ValueError(f"num_classes 必须大于 0，当前={self.num_classes}")

        self.class_names = _normalize_class_names(class_names, self.num_classes)

        self.conf_thres = float(conf_thres)
        self.pr_iou_thres = float(pr_iou_thres)

        if not 0.0 <= self.conf_thres <= 1.0:
            raise ValueError(f"conf_thres 必须在 [0,1]，当前={self.conf_thres}")

        if not 0.0 < self.pr_iou_thres <= 1.0:
            raise ValueError(f"pr_iou_thres 必须在 (0,1]，当前={self.pr_iou_thres}")

        self.iou_thresholds = np.round(
            np.arange(float(map_iou_start), float(map_iou_end) + 1e-9, float(map_iou_step)),
            2,
        )

        if self.iou_thresholds.size == 0:
            raise ValueError("mAP IoU 阈值为空，请检查 map_iou_* 配置。")

        self.reset()

    def reset(self):

        self.pred_records = []
        self.gt_records = []
        self.image_id = 0

    def update(self, preds, targets):

        if len(preds) != len(targets):
            raise ValueError(f"preds 与 targets 数量不一致：{len(preds)} vs {len(targets)}")

        for pred, target in zip(preds, targets):
            image_id = self.image_id
            self.image_id += 1

            pred_boxes = to_numpy(pred.get("boxes"))
            pred_scores = to_numpy(pred.get("scores"))
            pred_labels = to_numpy(pred.get("labels"))

            gt_boxes = to_numpy(target.get("boxes"))
            gt_labels = to_numpy(target.get("labels"))

            if pred_boxes is None:
                pred_boxes = np.zeros((0, 4), dtype=np.float32)
            if pred_scores is None:
                pred_scores = np.zeros((0,), dtype=np.float32)
            if pred_labels is None:
                pred_labels = np.zeros((0,), dtype=np.int64)

            if gt_boxes is None:
                gt_boxes = np.zeros((0, 4), dtype=np.float32)
            if gt_labels is None:
                gt_labels = np.zeros((0,), dtype=np.int64)

            pred_boxes = pred_boxes.reshape(-1, 4).astype(np.float32)
            pred_scores = pred_scores.reshape(-1).astype(np.float32)
            pred_labels = pred_labels.reshape(-1).astype(np.int64)

            gt_boxes = gt_boxes.reshape(-1, 4).astype(np.float32)
            gt_labels = gt_labels.reshape(-1).astype(np.int64)

            if not (len(pred_boxes) == len(pred_scores) == len(pred_labels)):
                raise ValueError(
                    f"预测数量不一致：boxes={len(pred_boxes)}, "
                    f"scores={len(pred_scores)}, labels={len(pred_labels)}"
                )

            if len(gt_boxes) != len(gt_labels):
                raise ValueError(f"GT 数量不一致：boxes={len(gt_boxes)}, labels={len(gt_labels)}")

            for box, score, label in zip(pred_boxes, pred_scores, pred_labels):
                label = int(label)

                if 0 <= label < self.num_classes:
                    self.pred_records.append({
                        "image_id": image_id,
                        "box": box.astype(np.float32),
                        "score": float(score),
                        "label": label,
                    })

            for box, label in zip(gt_boxes, gt_labels):
                label = int(label)

                if 0 <= label < self.num_classes:
                    self.gt_records.append({
                        "image_id": image_id,
                        "box": box.astype(np.float32),
                        "label": label,
                    })

    def compute(self) -> Dict:

        per_class = {}

        total_tp = 0
        total_fp = 0
        total_fn = 0

        map50_list = []
        map50_95_list = []

        for class_id in range(self.num_classes):
            class_result = self._compute_one_class(class_id)
            per_class[class_id] = class_result

            total_tp += class_result["TP"]
            total_fp += class_result["FP"]
            total_fn += class_result["FN"]

            if class_result["num_gt"] > 0:
                map50_list.append(class_result["mAP50"])
                map50_95_list.append(class_result["mAP50_95"])

        precision = total_tp / (total_tp + total_fp) if total_tp + total_fp > 0 else 0.0
        recall = total_tp / (total_tp + total_fn) if total_tp + total_fn > 0 else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0

        map50 = float(np.mean(map50_list)) if map50_list else 0.0
        map50_95 = float(np.mean(map50_95_list)) if map50_95_list else 0.0

        return {
            "P": float(precision),
            "R": float(recall),
            "F1": float(f1),
            "mAP50": map50,
            "mAP50_95": map50_95,
            "per_class": per_class,
            "num_gt": int(len(self.gt_records)),
            "num_pred": int(len(self.pred_records)),
        }

    def _compute_one_class(self, class_id: int) -> Dict:

        class_preds = [pred for pred in self.pred_records if pred["label"] == class_id]
        class_gts = [gt for gt in self.gt_records if gt["label"] == class_id]

        num_gt = len(class_gts)

        tp, fp, fn = self._match_for_pr(
            class_preds,
            class_gts,
            iou_thres=self.pr_iou_thres,
            conf_thres=self.conf_thres,
        )

        precision = tp / (tp + fp) if tp + fp > 0 else 0.0
        recall = tp / (tp + fn) if tp + fn > 0 else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0

        aps = []

        for iou_thres in self.iou_thresholds:
            ap = self._compute_ap_one_class(class_preds, class_gts, float(iou_thres))
            aps.append(ap)

        map50 = float(aps[0]) if aps else 0.0
        map50_95 = float(np.mean(aps)) if aps else 0.0

        return {
            "name": self.class_names[class_id],
            "P": float(precision),
            "R": float(recall),
            "F1": float(f1),
            "mAP50": map50,
            "mAP50_95": map50_95,
            "TP": int(tp),
            "FP": int(fp),
            "FN": int(fn),
            "num_gt": int(num_gt),
            "num_pred": int(len(class_preds)),
        }

    def _match_for_pr(self, class_preds, class_gts, iou_thres, conf_thres):

        preds = [pred for pred in class_preds if pred["score"] >= conf_thres]
        preds = sorted(preds, key=lambda x: x["score"], reverse=True)

        gt_by_image = {}

        for gt in class_gts:
            gt_by_image.setdefault(gt["image_id"], []).append(gt)

        matched = {
            image_id: np.zeros(len(gts), dtype=bool)
            for image_id, gts in gt_by_image.items()
        }

        tp = 0
        fp = 0

        for pred in preds:
            image_id = pred["image_id"]
            gts = gt_by_image.get(image_id, [])

            if len(gts) == 0:
                fp += 1
                continue

            pred_box = pred["box"].reshape(1, 4)
            gt_boxes = np.stack([gt["box"] for gt in gts], axis=0)

            ious = box_iou(pred_box, gt_boxes)[0]

            ious[matched[image_id]] = -1.0

            best_index = int(np.argmax(ious))
            best_iou = float(ious[best_index])

            if best_iou >= iou_thres:
                tp += 1
                matched[image_id][best_index] = True
            else:
                fp += 1

        fn = sum(int((~flags).sum()) for flags in matched.values())

        return tp, fp, fn

    def _compute_ap_one_class(self, class_preds, class_gts, iou_thres):

        if len(class_gts) == 0 or len(class_preds) == 0:
            return 0.0

        preds = sorted(class_preds, key=lambda x: x["score"], reverse=True)

        gt_by_image = {}

        for gt in class_gts:
            gt_by_image.setdefault(gt["image_id"], []).append(gt)

        matched = {
            image_id: np.zeros(len(gts), dtype=bool)
            for image_id, gts in gt_by_image.items()
        }

        tp = np.zeros(len(preds), dtype=np.float32)
        fp = np.zeros(len(preds), dtype=np.float32)

        for index, pred in enumerate(preds):
            image_id = pred["image_id"]
            gts = gt_by_image.get(image_id, [])

            if len(gts) == 0:
                fp[index] = 1.0
                continue

            pred_box = pred["box"].reshape(1, 4)
            gt_boxes = np.stack([gt["box"] for gt in gts], axis=0)

            ious = box_iou(pred_box, gt_boxes)[0]

            ious[matched[image_id]] = -1.0

            best_index = int(np.argmax(ious))
            best_iou = float(ious[best_index])

            if best_iou >= iou_thres:
                tp[index] = 1.0
                matched[image_id][best_index] = True
            else:
                fp[index] = 1.0

        tp_cum = np.cumsum(tp)
        fp_cum = np.cumsum(fp)

        recall = tp_cum / max(len(class_gts), 1)
        precision = tp_cum / np.maximum(tp_cum + fp_cum, 1e-12)

        return compute_ap(recall, precision)

    def summary_string(self) -> str:

        result = self.compute()

        return (
            f"P={result['P']:.4f} | "
            f"R={result['R']:.4f} | "
            f"F1={result['F1']:.4f} | "
            f"mAP50={result['mAP50']:.4f} | "
            f"mAP50-95={result['mAP50_95']:.4f}"
        )

    def class_summary_string(self) -> str:

        result = self.compute()
        parts = []

        for class_id in range(self.num_classes):
            r = result["per_class"][class_id]

            parts.append(
                f"{r['name']}: "
                f"P={r['P']:.3f}, R={r['R']:.3f}, F1={r['F1']:.3f}, "
                f"mAP50={r['mAP50']:.3f}, mAP50-95={r['mAP50_95']:.3f}"
            )

        return " | ".join(parts)
