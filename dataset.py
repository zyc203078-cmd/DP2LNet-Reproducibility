from typing import Optional, Tuple

import numpy as np
import torch
from PIL import Image, ImageEnhance

class BridgeTransform:

    def __init__(
        self,
        img_size: int = 512,
        is_train: bool = True,
        aug_level: str = "none",
        normalize_mode: str = "zero_one",
    ):

        self.img_size = int(img_size)
        self.is_train = bool(is_train)
        self.aug_level = str(aug_level)
        self.normalize_mode = str(normalize_mode)

        if self.aug_level not in [
            "none",
            "medium",
            "strong",
        ]:
            raise ValueError(
                f"不支持 aug_level={self.aug_level}"
            )

        if self.normalize_mode not in [
            "zero_one",
            "imagenet",
            "none",
        ]:
            raise ValueError(
                f"不支持 normalize_mode={self.normalize_mode}"
            )

    def __call__(
        self,
        image: Image.Image,
        mask: Optional[np.ndarray],
        boxes: np.ndarray,
    ) -> Tuple[
        torch.Tensor,
        Optional[torch.Tensor],
        torch.Tensor,
    ]:

        orig_w, orig_h = image.size

        if mask is not None:
            if mask.shape[:2] != (
                orig_h,
                orig_w,
            ):
                raise ValueError(
                    f"Mask 尺寸 {mask.shape[:2]} "
                    f"与图像尺寸 {(orig_h, orig_w)} 不一致。"
                )

        boxes = boxes.astype(
            np.float32
        ).copy()

        image = image.resize(
            (
                self.img_size,
                self.img_size,
            ),
            Image.BILINEAR,
        )

        if mask is not None:

            mask_image = Image.fromarray(
                mask.astype(np.uint8)
            )

            mask_image = mask_image.resize(
                (
                    self.img_size,
                    self.img_size,
                ),
                Image.NEAREST,
            )

            mask = np.asarray(
                mask_image,
                dtype=np.int64,
            )

        if boxes.size > 0:

            scale_x = (
                self.img_size
                / float(orig_w)
            )

            scale_y = (
                self.img_size
                / float(orig_h)
            )

            boxes[:, [0, 2]] *= scale_x
            boxes[:, [1, 3]] *= scale_y

            boxes = self._clip_boxes(
                boxes
            )

        if self.is_train:

            image, mask, boxes = self._augment(
                image,
                mask,
                boxes,
            )

        image_tensor = self._image_to_tensor(
            image
        )

        if mask is not None:

            mask_tensor = torch.from_numpy(
                mask.astype(np.int64)
            )

        else:

            mask_tensor = None

        boxes_tensor = torch.from_numpy(
            boxes.astype(np.float32)
        )

        return (
            image_tensor,
            mask_tensor,
            boxes_tensor,
        )

    def _augment(
        self,
        image,
        mask,
        boxes,
    ):

        if self.aug_level == "none":
            return image, mask, boxes

        if np.random.rand() < 0.5:

            image = image.transpose(
                Image.Transpose.FLIP_LEFT_RIGHT
            )

            if mask is not None:

                mask = np.ascontiguousarray(
                    mask[:, ::-1]
                )

            if boxes.size > 0:

                x1 = boxes[:, 0].copy()
                x2 = boxes[:, 2].copy()

                boxes[:, 0] = (
                    self.img_size - x2
                )

                boxes[:, 2] = (
                    self.img_size - x1
                )

                boxes = self._clip_boxes(
                    boxes
                )

        if self.aug_level in [
            "medium",
            "strong",
        ]:

            if np.random.rand() < 0.5:

                factor = np.random.uniform(
                    0.85,
                    1.15,
                )

                image = (
                    ImageEnhance.Brightness(
                        image
                    ).enhance(
                        factor
                    )
                )

            if np.random.rand() < 0.5:

                factor = np.random.uniform(
                    0.85,
                    1.15,
                )

                image = (
                    ImageEnhance.Contrast(
                        image
                    ).enhance(
                        factor
                    )
                )

        if self.aug_level == "strong":

            if np.random.rand() < 0.3:

                factor = np.random.uniform(
                    0.85,
                    1.15,
                )

                image = (
                    ImageEnhance.Color(
                        image
                    ).enhance(
                        factor
                    )
                )

        return image, mask, boxes

    def _image_to_tensor(
        self,
        image: Image.Image,
    ) -> torch.Tensor:

        array = np.asarray(
            image,
            dtype=np.float32,
        )

        if self.normalize_mode == "zero_one":

            array = (
                array / 255.0
            )

        elif self.normalize_mode == "imagenet":

            array = (
                array / 255.0
            )

            mean = np.asarray(
                [
                    0.485,
                    0.456,
                    0.406,
                ],
                dtype=np.float32,
            )

            std = np.asarray(
                [
                    0.229,
                    0.224,
                    0.225,
                ],
                dtype=np.float32,
            )

            array = (
                array - mean
            ) / std

        elif self.normalize_mode == "none":
            pass

        array = array.transpose(
            2,
            0,
            1,
        )

        return torch.from_numpy(
            array.copy()
        ).float()

    def _clip_boxes(
        self,
        boxes: np.ndarray,
    ) -> np.ndarray:

        if boxes.size == 0:

            return boxes.reshape(
                0,
                4,
            ).astype(
                np.float32
            )

        boxes[:, 0::2] = np.clip(
            boxes[:, 0::2],
            0.0,
            float(self.img_size - 1),
        )

        boxes[:, 1::2] = np.clip(
            boxes[:, 1::2],
            0.0,
            float(self.img_size - 1),
        )

        return boxes.astype(
            np.float32
        )


from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image


IMG_EXTS = [
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
]

class BridgeDataset(torch.utils.data.Dataset):

    def __init__(
        self,
        image_dir: str,
        mask_dir: Optional[str] = None,
        label_dir: Optional[str] = None,
        img_size: int = 512,
        is_train: bool = True,
        aug_level: str = "none",
        normalize_mode: str = "zero_one",
        mask_encoding: str = "index",
        task_type: str = "multitask",
        seg_num_classes: Optional[int] = None,
        det_num_classes: Optional[int] = None,
        seg_class_names: Optional[List[str]] = None,
        det_class_names: Optional[List[str]] = None,
    ):

        self.image_dir = Path(
            image_dir
        )

        self.mask_dir = (
            Path(mask_dir)
            if mask_dir not in [
                None,
                "",
                "None",
                "null",
            ]
            else None
        )

        self.label_dir = (
            Path(label_dir)
            if label_dir not in [
                None,
                "",
                "None",
                "null",
            ]
            else None
        )

        self.img_size = int(
            img_size
        )

        self.is_train = bool(
            is_train
        )

        self.aug_level = str(
            aug_level
        )

        self.normalize_mode = str(
            normalize_mode
        )

        self.mask_encoding = str(
            mask_encoding
        ).lower()

        self.task_type = str(
            task_type
        ).lower()

        if seg_num_classes is None:

            raise ValueError(
                "必须显式传入 seg_num_classes。"
            )

        if det_num_classes is None:

            raise ValueError(
                "必须显式传入 det_num_classes。"
            )

        self.seg_num_classes = int(
            seg_num_classes
        )

        self.det_num_classes = int(
            det_num_classes
        )

        self.seg_class_names = (
            list(seg_class_names)
            if seg_class_names is not None
            else None
        )

        self.det_class_names = (
            list(det_class_names)
            if det_class_names is not None
            else None
        )

        self._validate_config()

        self.image_paths = (
            self._collect_images(
                self.image_dir
            )
        )

        if len(self.image_paths) == 0:

            raise RuntimeError(
                f"没有在目录中找到图像："
                f"{self.image_dir}"
            )

        self.transform = BridgeTransform(
            img_size=self.img_size,
            is_train=self.is_train,
            aug_level=self.aug_level,
            normalize_mode=self.normalize_mode,
        )

        self._print_info()

    def __len__(self):

        return len(
            self.image_paths
        )

    def __getitem__(
        self,
        index: int,
    ) -> Dict:

        image_path = self.image_paths[
            index
        ]

        stem = image_path.stem

        image = Image.open(
            image_path
        ).convert(
            "RGB"
        )

        orig_w, orig_h = (
            image.size
        )

        mask_path = (
            self._find_mask_path(
                stem
            )
        )

        label_path = (
            self._find_label_path(
                stem
            )
        )

        has_seg = (
            mask_path is not None
        )

        has_det = (
            label_path is not None
        )

        if (
            self.task_type
            == "segmentation"
        ):
            has_det = False

        if (
            self.task_type
            == "detection"
        ):
            has_seg = False

        if has_seg:

            mask = self._load_mask(
                mask_path
            )

        else:

            mask = np.zeros(
                (
                    orig_h,
                    orig_w,
                ),
                dtype=np.int64,
            )

        if has_det:

            boxes, labels = (
                self._load_yolo_labels(
                    label_path=label_path,
                    image_w=orig_w,
                    image_h=orig_h,
                )
            )

        else:

            boxes = np.zeros(
                (0, 4),
                dtype=np.float32,
            )

            labels = np.zeros(
                (0,),
                dtype=np.int64,
            )

        (
            image_tensor,
            mask_tensor,
            boxes_tensor,
        ) = self.transform(
            image=image,
            mask=mask,
            boxes=boxes,
        )

        if boxes_tensor.numel() == 0:

            labels_tensor = torch.zeros(
                (0,),
                dtype=torch.long,
            )

        else:

            labels_tensor = torch.as_tensor(
                labels,
                dtype=torch.long,
            )

            if (
                labels_tensor.shape[0]
                != boxes_tensor.shape[0]
            ):

                raise ValueError(
                    f"Transform 后 boxes 与 labels 数量不一致："
                    f"image={image_path.name}, "
                    f"boxes={boxes_tensor.shape[0]}, "
                    f"labels={labels_tensor.shape[0]}"
                )

        sample = {

            "image": image_tensor,

            "mask": mask_tensor.long(),

            "masks": mask_tensor.long(),

            "boxes": boxes_tensor.float(),
            "labels": labels_tensor.long(),

            "has_seg": bool(
                has_seg
            ),

            "has_det": bool(
                has_det
            ),

            "image_id": stem,
            "id": stem,

            "image_path": str(
                image_path
            ),

            "mask_path": (
                str(mask_path)
                if mask_path is not None
                else ""
            ),

            "label_path": (
                str(label_path)
                if label_path is not None
                else ""
            ),

            "orig_size": (
                orig_h,
                orig_w,
            ),

            "size": (
                self.img_size,
                self.img_size,
            ),
        }

        return sample

    def _validate_config(self):

        if self.task_type not in [
            "segmentation",
            "detection",
            "multitask",
        ]:

            raise ValueError(
                f"不支持 task_type="
                f"{self.task_type}"
            )

        if self.seg_num_classes <= 1:

            raise ValueError(
                "seg_num_classes "
                "必须大于 1。"
            )

        if self.det_num_classes <= 0:

            raise ValueError(
                "det_num_classes "
                "必须大于 0。"
            )

        if (
            self.seg_class_names
            is not None
            and len(
                self.seg_class_names
            )
            != self.seg_num_classes
        ):

            raise ValueError(
                "seg_class_names 数量 "
                "与 seg_num_classes 不一致。"
            )

        if (
            self.det_class_names
            is not None
            and len(
                self.det_class_names
            )
            != self.det_num_classes
        ):

            raise ValueError(
                "det_class_names 数量 "
                "与 det_num_classes 不一致。"
            )

        if not self.image_dir.exists():

            raise FileNotFoundError(
                f"image_dir 不存在："
                f"{self.image_dir}"
            )

        if self.task_type in [
            "segmentation",
            "multitask",
        ]:

            if self.mask_dir is None:

                raise ValueError(
                    "当前任务需要 mask_dir。"
                )

        if (
            self.mask_dir is not None
            and not self.mask_dir.exists()
        ):

            raise FileNotFoundError(
                f"mask_dir 不存在："
                f"{self.mask_dir}"
            )

        if self.task_type in [
            "detection",
            "multitask",
        ]:

            if self.label_dir is None:

                raise ValueError(
                    "当前任务需要 label_dir。"
                )

        if (
            self.label_dir is not None
            and not self.label_dir.exists()
        ):

            raise FileNotFoundError(
                f"label_dir 不存在："
                f"{self.label_dir}"
            )

        if self.mask_encoding not in [
            "index",
            "contiguous",
        ]:

            raise ValueError(
                "当前精简版本只支持 "
                "index 类型的分割 mask。"
            )

    @staticmethod
    def _collect_images(
        image_dir: Path,
    ) -> List[Path]:

        image_paths = []

        for ext in IMG_EXTS:

            image_paths.extend(
                image_dir.glob(
                    f"*{ext}"
                )
            )

            image_paths.extend(
                image_dir.glob(
                    f"*{ext.upper()}"
                )
            )

        image_paths = sorted(
            set(image_paths),
            key=lambda path: path.name,
        )

        return image_paths

    def _find_mask_path(
        self,
        stem: str,
    ) -> Optional[Path]:

        if self.mask_dir is None:
            return None

        for ext in IMG_EXTS:

            path = (
                self.mask_dir
                / f"{stem}{ext}"
            )

            if path.exists():
                return path

            path_upper = (
                self.mask_dir
                / f"{stem}{ext.upper()}"
            )

            if path_upper.exists():
                return path_upper

        return None

    def _find_label_path(
        self,
        stem: str,
    ) -> Optional[Path]:

        if self.label_dir is None:
            return None

        path = (
            self.label_dir
            / f"{stem}.txt"
        )

        if path.exists():
            return path

        return None

    def _load_mask(
        self,
        mask_path: Path,
    ) -> np.ndarray:

        mask = np.array(
            Image.open(
                mask_path
            )
        )

        if mask.ndim != 2:

            raise ValueError(
                f"Mask 必须是单通道索引图："
                f"{mask_path}, "
                f"shape={mask.shape}"
            )

        mask = mask.astype(
            np.int64
        )

        if mask.size > 0:

            min_value = int(
                mask.min()
            )

            max_value = int(
                mask.max()
            )

            if (
                min_value < 0
                or max_value
                >= self.seg_num_classes
            ):

                unique_values = np.unique(
                    mask
                ).tolist()

                raise ValueError(
                    f"Mask 类别索引越界："
                    f"{mask_path}\n"
                    f"当前值={unique_values}\n"
                    f"允许范围="
                    f"[0, {self.seg_num_classes - 1}]"
                )

        return mask

    def _load_yolo_labels(
        self,
        label_path: Path,
        image_w: int,
        image_h: int,
    ) -> Tuple[
        np.ndarray,
        np.ndarray,
    ]:

        boxes = []
        labels = []

        with open(
            label_path,
            "r",
            encoding="utf-8",
        ) as file:

            lines = file.readlines()

        for line_index, line in enumerate(
            lines,
            start=1,
        ):

            line = line.strip()

            if line == "":
                continue

            parts = line.split()

            if len(parts) != 5:

                raise ValueError(
                    f"YOLO 标签格式错误："
                    f"{label_path}, "
                    f"line={line_index}"
                )

            try:

                cls_id = int(
                    float(
                        parts[0]
                    )
                )

                cx = float(
                    parts[1]
                )

                cy = float(
                    parts[2]
                )

                box_w = float(
                    parts[3]
                )

                box_h = float(
                    parts[4]
                )

            except ValueError:

                raise ValueError(
                    f"YOLO 标签数值错误："
                    f"{label_path}, "
                    f"line={line_index}"
                )

            if (
                cls_id < 0
                or cls_id
                >= self.det_num_classes
            ):

                raise ValueError(
                    f"检测类别越界："
                    f"{label_path}, "
                    f"class={cls_id}"
                )

            if not (
                0.0 <= cx <= 1.0
                and 0.0 <= cy <= 1.0
                and 0.0 < box_w <= 1.0
                and 0.0 < box_h <= 1.0
            ):

                raise ValueError(
                    f"YOLO 坐标错误："
                    f"{label_path}, "
                    f"line={line_index}"
                )

            x1 = (
                cx - box_w / 2.0
            ) * image_w

            y1 = (
                cy - box_h / 2.0
            ) * image_h

            x2 = (
                cx + box_w / 2.0
            ) * image_w

            y2 = (
                cy + box_h / 2.0
            ) * image_h

            x1 = np.clip(
                x1,
                0.0,
                image_w - 1.0,
            )

            y1 = np.clip(
                y1,
                0.0,
                image_h - 1.0,
            )

            x2 = np.clip(
                x2,
                0.0,
                image_w - 1.0,
            )

            y2 = np.clip(
                y2,
                0.0,
                image_h - 1.0,
            )

            if (
                x2 <= x1
                or y2 <= y1
            ):

                raise ValueError(
                    f"检测框无效："
                    f"{label_path}, "
                    f"line={line_index}"
                )

            boxes.append(
                [
                    x1,
                    y1,
                    x2,
                    y2,
                ]
            )

            labels.append(
                cls_id
            )

        if len(boxes) == 0:

            return (
                np.zeros(
                    (0, 4),
                    dtype=np.float32,
                ),
                np.zeros(
                    (0,),
                    dtype=np.int64,
                ),
            )

        return (
            np.asarray(
                boxes,
                dtype=np.float32,
            ),
            np.asarray(
                labels,
                dtype=np.int64,
            ),
        )

    def _print_info(self):

        print(
            "=" * 80
        )

        print(
            "[BridgeDataset]"
        )

        print(
            f"task_type       : "
            f"{self.task_type}"
        )

        print(
            f"image_dir       : "
            f"{self.image_dir}"
        )

        print(
            f"mask_dir        : "
            f"{self.mask_dir}"
        )

        print(
            f"label_dir       : "
            f"{self.label_dir}"
        )

        print(
            f"num_images      : "
            f"{len(self.image_paths)}"
        )

        print(
            f"img_size        : "
            f"{self.img_size}"
        )

        print(
            f"is_train        : "
            f"{self.is_train}"
        )

        print(
            f"aug_level       : "
            f"{self.aug_level}"
        )

        print(
            f"normalize_mode  : "
            f"{self.normalize_mode}"
        )

        print(
            f"seg_num_classes : "
            f"{self.seg_num_classes}"
        )

        print(
            f"det_num_classes : "
            f"{self.det_num_classes}"
        )

        if (
            self.seg_class_names
            is not None
        ):

            print(
                f"seg_class_names : "
                f"{self.seg_class_names}"
            )

        if (
            self.det_class_names
            is not None
        ):

            print(
                f"det_class_names : "
                f"{self.det_class_names}"
            )

        print(
            "=" * 80
        )


import torch

def bridge_collate_fn(batch):

    images = torch.stack(
        [
            item["image"]
            for item in batch
        ],
        dim=0,
    )

    masks = torch.stack(
        [
            item["mask"]
            for item in batch
        ],
        dim=0,
    )

    boxes = [
        item["boxes"]
        for item in batch
    ]

    labels = [
        item["labels"]
        for item in batch
    ]

    has_seg = [
        item["has_seg"]
        for item in batch
    ]

    has_det = [
        item["has_det"]
        for item in batch
    ]

    image_ids = [
        item["image_id"]
        for item in batch
    ]

    image_paths = [
        item["image_path"]
        for item in batch
    ]

    mask_paths = [
        item["mask_path"]
        for item in batch
    ]

    label_paths = [
        item["label_path"]
        for item in batch
    ]

    orig_sizes = [
        item["orig_size"]
        for item in batch
    ]

    sizes = [
        item["size"]
        for item in batch
    ]

    return {
        "images": images,
        "masks": masks,

        "boxes": boxes,
        "labels": labels,

        "has_seg": has_seg,
        "has_det": has_det,

        "image_ids": image_ids,

        "image_paths": image_paths,
        "mask_paths": mask_paths,
        "label_paths": label_paths,

        "orig_sizes": orig_sizes,
        "sizes": sizes,
    }
