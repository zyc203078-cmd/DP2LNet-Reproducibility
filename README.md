# DP2LNet

Minimal training and evaluation code for DP2LNet.

## Environment

- Python 3.10
- PyTorch 2.2.2
- CUDA 11.8

```bash
pip install -r requirements.txt
```

## Data layout

```text
data/
├── train/
│   ├── images/
│   ├── masks/
│   └── labels/
├── val/
│   ├── images/
│   ├── masks/
│   └── labels/
└── test/
    ├── images/
    ├── masks/
    └── labels/
```

Segmentation classes: `BG`, `CR`, `CO`, `SP`.
Detection classes: `EC`, `DB`, `EB`.
Detection annotations use YOLO format: `class cx cy w h`.

The complete UAV image dataset and trained weights are not distributed because of data access restrictions.

## Training

The repeated experiments use random seeds `42`, `3407`, and `2026`.

```bash
python train.py --config config.yaml --seed 42 --device cuda:0
python train.py --config config.yaml --seed 3407 --device cuda:0
python train.py --config config.yaml --seed 2026 --device cuda:0
```

## Evaluation

```bash
python evaluate.py --config config.yaml --weights /path/to/best.pth --split test --device cuda:0
```
