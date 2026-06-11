'''Training pipeline entry point.

Trains a DINOv2/DINOv3 backbone with a classification head, logging
loss, accuracy and macro F1 per epoch, and saving checkpoints plus a
history.json with the full metric history.

Usage:
    python train.py --config configs/baseline.yaml
'''
import argparse

import torch
import torch.nn as nn

from train.model import build_model_from_config
from train.trainer import Trainer
from utils.config_utils import load_config
from utils.data_utils import build_train_loader, build_eval_loader


# ------------------------------------------------------------------
# Builder helpers
# ------------------------------------------------------------------

_OPTIMIZERS = {
    'adam':  torch.optim.Adam,
    'adamw': torch.optim.AdamW,
    'sgd':   torch.optim.SGD,
}

_LOSSES = {
    'cross_entropy': nn.CrossEntropyLoss,
}


def build_optimizer(name: str, params, learning_rate: float) -> torch.optim.Optimizer:
    return _OPTIMIZERS[name](params, lr=learning_rate)


def build_loss(name: str) -> nn.Module:
    return _LOSSES[name]()


# ------------------------------------------------------------------
# Main pipeline
# ------------------------------------------------------------------

def main(config_path: str) -> None:
    cfg = load_config(config_path)

    training  = cfg['training']
    model_cfg = cfg['model']

    if model_cfg['head']['type'] == 'none':
        raise ValueError(
            "head.type='none' is feature-extractor mode and cannot be trained "
            "with a classification loss. Use extract_features.py instead, or "
            "set head.type to 'linear'/'mlp'."
        )

    # Build model
    model = build_model_from_config(model_cfg)

    # Only pass parameters that require gradients to the optimizer
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = build_optimizer(training['optimizer'], trainable_params, training['learning_rate'])
    loss_fn   = build_loss(training['loss_function'])

    # Dataloaders
    dataset_cfg  = cfg['dataset']
    batch_size   = training['batch_size']
    train_loader = build_train_loader(dataset_cfg['train_path'], batch_size)
    val_loader   = build_eval_loader(dataset_cfg['val_path'], batch_size)
    print(
        f"Dataset → train: {len(train_loader.dataset)} images | "
        f"val: {len(val_loader.dataset)} images | "
        f"classes: {train_loader.dataset.classes}"
    )

    # Training
    checkpoints_dir = training.get('checkpoints_dir') or 'checkpoints'
    trainer = Trainer(model, optimizer, loss_fn, training['device'], checkpoints_dir)
    trainer.train(train_loader, val_loader, training['epochs'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DINOv2/DINOv3 training pipeline')
    parser.add_argument('--config', type=str, required=True, help='Path to the YAML configuration file')
    args = parser.parse_args()

    main(args.config)
