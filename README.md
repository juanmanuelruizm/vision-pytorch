# vision-pytorch

![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)
![Status](https://img.shields.io/badge/Status-In%20Development-yellow)

Modular PyTorch pipeline for training and fine-tuning vision models based on DINOv2 (Meta) backbones over ViT architectures.

## Overview

The pipeline loads a DINOv2 ViT backbone from a local `.pth` file (Meta-provided weights), optionally freezes a configurable fraction of it, attaches a classification head, and runs a full train/validation loop with checkpoint management.

## Project Structure

```
vision-pytorch/
├── train.py                  # Entry point — reads config and launches training
├── configs/
│   └── baseline.yaml         # Full configuration template
├── train/
│   ├── model.py              # Model: DINOv2 backbone + classification head
│   └── trainer.py            # Training/validation loop + checkpoint saving
└── utils/
    ├── model_utils.py        # Backbone loading and freeze logic
    ├── head_utils.py         # Classification head factory (linear / mlp)
    └── config_utils.py       # YAML loading and validation
```

## Supported Backbone Variants

| Variant | Hub name | Embed dim |
|---|---|---|
| `vit_small` | `dinov2_vits14` | 384 |
| `vit_base` | `dinov2_vitb14` | 768 |
| `vit_large` | `dinov2_vitl14` | 1024 |
| `vit_giant` | `dinov2_vitg14` | 1536 |

## Configuration

All training parameters are defined in a YAML file. Fill in `configs/baseline.yaml`:

```yaml
training:
  epochs: 50
  batch_size: 32
  learning_rate: 0.0001
  optimizer: adam           # adam | sgd
  loss_function: cross_entropy
  device: cuda              # cuda | cpu
  checkpoints_dir: checkpoints/experiment_01

model:
  variant: vit_base
  weights_path: weights/dinov2_vitb14.pth
  num_classes: 10
  freeze_ratio: 1.0         # 0.0 (full fine-tuning) → 1.0 (linear probe)

  head:
    type: linear            # linear | mlp
    hidden_dim: null        # mlp only
    dropout: null           # mlp only

dataset:
  train_path: data/train
  val_path: data/val
```

**`freeze_ratio`** controls how much of the backbone is frozen:
- `0.0` — nothing frozen, full fine-tuning
- `1.0` — backbone fully frozen, only the head trains (linear probe)
- `0.5` — embedding layers + first 50% of transformer blocks frozen

## Usage

```bash
pip install -r requirements.txt
python train.py --config configs/baseline.yaml
```

The dataset is expected in [`ImageFolder`](https://pytorch.org/vision/stable/generated/torchvision.datasets.ImageFolder.html) format (one subfolder per class):

```
data/
├── train/
│   ├── class_a/
│   └── class_b/
└── val/
    ├── class_a/
    └── class_b/
```

## Checkpoints

At the end of each epoch the trainer saves:
- `checkpoints_dir/checkpoint_epoch_NNN.pth` — epoch checkpoint
- `checkpoints_dir/best_model.pth` — best model by validation accuracy

## Requirements

- Python 3.10+
- torch >= 2.0.0
- torchvision >= 0.15.0
- pyyaml >= 6.0
