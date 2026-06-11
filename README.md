# vision-pytorch

![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat&logo=python&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=flat)
![Status](https://img.shields.io/badge/Status-In%20Development-yellow)

Modular PyTorch pipelines for training, evaluating and using **DINOv2 / DINOv3** (Meta) ViT backbones — as fine-tuned classifiers, frozen backbones with classification heads, or raw feature extractors.

## Overview

The project loads a DINO ViT backbone from a local `.pth` file (Meta-provided weights) and offers three independent pipelines:

| Pipeline | Entry point | What it does |
|---|---|---|
| **Train** | `train.py` | Fine-tunes backbone + head (or just the head), logging loss, accuracy and macro F1 per epoch, with checkpoints and a `history.json` metric history. |
| **Evaluate** | `evaluate.py` | Loads a trained checkpoint and evaluates it on the **test set**: loss, accuracy, top-k accuracy, per-class precision/recall/F1 (macro & weighted) and confusion matrix. Saves metrics as JSON. |
| **Extract features** | `extract_features.py` | Runs the backbone *as-is* (no head) over a dataset and dumps embeddings to a `.pt` file — for k-NN/linear probes, clustering or retrieval. |

Since DINO models are foremost **feature extractors**, the backbone can be used in three modes:
- **Full fine-tuning** — `freeze_ratio: 0.0` + a `linear`/`mlp` head.
- **Linear probe** — `freeze_ratio: 1.0` (frozen backbone) + a head; only the head trains.
- **Raw feature extractor** — `head.type: none`; the model outputs `[B, embed_dim]` embeddings directly (use `extract_features.py`).

## Project Structure

```
vision-pytorch/
├── train.py                  # Training pipeline entry point
├── evaluate.py               # Test evaluation pipeline entry point (separate from training)
├── extract_features.py       # Feature extraction pipeline entry point (backbone as-is)
├── configs/
│   └── baseline.yaml         # Full configuration template
├── train/
│   ├── model.py              # Model: DINO backbone + optional classification head
│   └── trainer.py            # Training/validation loop + metrics + checkpoints
├── evaluation/
│   └── evaluator.py          # Test evaluation loop + metrics report
└── utils/
    ├── model_utils.py        # Backbone families/variants, loading and freeze logic
    ├── head_utils.py         # Classification head factory (linear / mlp / none)
    ├── metrics.py            # MetricsTracker: accuracy, top-k, P/R/F1, confusion matrix
    ├── data_utils.py         # Shared transforms and dataloaders (train / eval)
    └── config_utils.py       # YAML loading and validation
```

## Supported Backbones

### DINOv3 (`family: dinov3`, patch size 16)

| Variant | Hub name | Embed dim |
|---|---|---|
| `vit_small` | `dinov3_vits16` | 384 |
| `vit_small_plus` | `dinov3_vits16plus` | 384 |
| `vit_base` | `dinov3_vitb16` | 768 |
| `vit_large` | `dinov3_vitl16` | 1024 |
| `vit_huge_plus` | `dinov3_vith16plus` | 1280 |
| `vit_7b` | `dinov3_vit7b16` | 4096 |

### DINOv2 (`family: dinov2`, patch size 14)

| Variant | Hub name | Embed dim |
|---|---|---|
| `vit_small` | `dinov2_vits14` | 384 |
| `vit_base` | `dinov2_vitb14` | 768 |
| `vit_large` | `dinov2_vitl14` | 1024 |
| `vit_giant` | `dinov2_vitg14` | 1536 |

## Obtaining the Weights

Weights are **never downloaded automatically**: the architecture is instantiated from `torch.hub` without pretrained weights and your local `.pth` file is loaded on top.

- **DINOv3** — request access and download the checkpoints from the [facebookresearch/dinov3](https://github.com/facebookresearch/dinov3) repository (Meta distributes them under their license after acceptance). Point `weights_path` at the downloaded file, e.g. `weights/dinov3_vitb16_pretrain_lvd1689m.pth`.
- **DINOv2** — direct downloads from [facebookresearch/dinov2](https://github.com/facebookresearch/dinov2), e.g. [dinov2_vitb14_pretrain.pth](https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth).

> Note: the first run needs internet access for `torch.hub` to fetch the *architecture code* (not the weights) from GitHub; it is cached afterwards.

## Configuration

All parameters are defined in a YAML file. Fill in `configs/baseline.yaml`:

```yaml
training:
  epochs: 50
  batch_size: 32
  learning_rate: 0.0001
  optimizer: adamw          # adam | adamw | sgd
  loss_function: cross_entropy
  device: cuda              # cuda | cpu
  checkpoints_dir: checkpoints/experiment_01

model:
  family: dinov3            # dinov2 | dinov3
  variant: vit_base         # see tables above (per family)
  weights_path: weights/dinov3_vitb16_pretrain_lvd1689m.pth
  num_classes: 10           # not required when head.type=none
  freeze_ratio: 1.0         # 0.0 (full fine-tuning) → 1.0 (linear probe)

  head:
    type: linear            # linear | mlp | none (none = raw feature extractor)
    hidden_dim: null        # mlp only
    dropout: null           # mlp only

dataset:
  train_path: data/train
  val_path: data/val
  test_path: data/test      # used by evaluate.py

evaluation:
  topk: [1, 5]              # top-k accuracies reported by evaluate.py
```

**`freeze_ratio`** controls how much of the backbone is frozen:
- `0.0` — nothing frozen, full fine-tuning
- `1.0` — backbone fully frozen, only the head trains (linear probe)
- `0.5` — embedding layers + first 50% of transformer blocks frozen

**`head.type`** selects the head:
- `linear` — single linear layer; fast, works well as linear probe
- `mlp` — Linear → GELU → Dropout → Linear; better for full fine-tuning
- `none` — no head: the model is a pure feature extractor (cannot be trained/evaluated as a classifier; use `extract_features.py`)

## Usage

```bash
pip install -r requirements.txt
```

### 1. Train

```bash
python train.py --config configs/baseline.yaml
```

Per epoch the trainer logs train loss/accuracy and validation loss, accuracy and macro F1, saves `checkpoint_epoch_NNN.pth`, keeps `best_model.pth` (by validation accuracy) and writes the full metric history to `checkpoints_dir/history.json`.

### 2. Evaluate on the test set

```bash
python evaluate.py --config configs/baseline.yaml --checkpoint checkpoints/experiment_01/best_model.pth
```

Prints a full report (loss, accuracy, top-k, per-class precision/recall/F1, macro & weighted averages) and saves the metrics — including the confusion matrix — to `test_metrics.json` next to the checkpoint. Options:

```bash
--data-path data/val            # evaluate a different split (overrides dataset.test_path)
--output results/metrics.json   # custom metrics path
--batch-size 64
```

### 3. Extract features (backbone as-is)

```bash
# From the raw Meta weights
python extract_features.py --config configs/baseline.yaml \
    --data-path data/train --output features/train.pt

# From a fine-tuned checkpoint
python extract_features.py --config configs/baseline.yaml \
    --data-path data/test --output features/test.pt \
    --checkpoint checkpoints/experiment_01/best_model.pth
```

The output `.pt` contains `{'features': [N, embed_dim], 'labels': [N], 'paths': [...], 'classes': [...]}`.

### Dataset format

Datasets are expected in [`ImageFolder`](https://pytorch.org/vision/stable/generated/torchvision.datasets.ImageFolder.html) format (one subfolder per class):

```
data/
├── train/
│   ├── class_a/
│   └── class_b/
├── val/
│   ├── class_a/
│   └── class_b/
└── test/
    ├── class_a/
    └── class_b/
```

## Requirements

- Python 3.10+
- torch >= 2.0.0
- torchvision >= 0.15.0
- pyyaml >= 6.0

## License

This project is licensed under the MIT License. Feel free to use, modify, and distribute it.

## Author

**Juan Manuel Ruiz Muñoz**

- LinkedIn: [Juan Manuel Ruiz Muñoz](https://www.linkedin.com/in/juan-manuel-ruiz-mu%C3%B1oz/)
- GitHub: [@juanmanuelruizm](https://github.com/juanmanuelruizm)
