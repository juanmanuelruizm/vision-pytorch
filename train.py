'''Training entry point.

Usage:
    python train.py --config configs/baseline.yaml
'''
import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from train.model import Model
from train.trainer import Trainer
from utils.config_utils import load_config


# ------------------------------------------------------------------
# Image transforms for DINOv2
# DINOv2 uses patch size 14; 224 = 16×14 (divisible by 14).
# Normalization with ImageNet statistics.
# ------------------------------------------------------------------
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD  = (0.229, 0.224, 0.225)

TRANSFORM_TRAIN = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
])

TRANSFORM_VAL = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
])

# ------------------------------------------------------------------
# Builder helpers
# ------------------------------------------------------------------

_OPTIMIZERS = {
    'adam': torch.optim.Adam,
    'sgd':  torch.optim.SGD,
}

_LOSSES = {
    'cross_entropy': nn.CrossEntropyLoss,
}


def build_optimizer(name: str, params, learning_rate: float) -> torch.optim.Optimizer:
    return _OPTIMIZERS[name](params, lr=learning_rate)


def build_loss(name: str) -> nn.Module:
    return _LOSSES[name]()


def build_dataloaders(cfg: dict) -> tuple[DataLoader, DataLoader]:
    dataset_cfg = cfg['dataset']
    batch_size  = cfg['training']['batch_size']

    train_ds = datasets.ImageFolder(dataset_cfg['train_path'], transform=TRANSFORM_TRAIN)
    val_ds   = datasets.ImageFolder(dataset_cfg['val_path'],   transform=TRANSFORM_VAL)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    print(f"Dataset → train: {len(train_ds)} images | val: {len(val_ds)} images | classes: {train_ds.classes}")
    return train_loader, val_loader


# ------------------------------------------------------------------
# Main pipeline
# ------------------------------------------------------------------

def main(config_path: str) -> None:
    cfg = load_config(config_path)

    training   = cfg['training']
    model_cfg  = cfg['model']
    head_cfg   = model_cfg['head']

    # Build model
    head_kwargs = {}
    if head_cfg['type'] == 'mlp':
        head_kwargs['hidden_dim'] = head_cfg['hidden_dim']
        head_kwargs['dropout']    = head_cfg.get('dropout', 0.0)

    model = Model(
        variant      = model_cfg['variant'],
        weights_path = model_cfg['weights_path'],
        num_classes  = model_cfg['num_classes'],
        head_type    = head_cfg['type'],
        freeze_ratio = float(model_cfg['freeze_ratio']),
        **head_kwargs,
    )

    # Only pass parameters that require gradients to the optimizer
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = build_optimizer(training['optimizer'], trainable_params, training['learning_rate'])
    loss_fn   = build_loss(training['loss_function'])

    # Dataloaders
    train_loader, val_loader = build_dataloaders(cfg)

    # Training
    checkpoints_dir = training.get('checkpoints_dir', 'checkpoints')
    trainer = Trainer(model, optimizer, loss_fn, training['device'], checkpoints_dir)
    trainer.train(train_loader, val_loader, training['epochs'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DINOv2 training pipeline')
    parser.add_argument('--config', type=str, required=True, help='Path to the YAML configuration file')
    args = parser.parse_args()

    main(args.config)
