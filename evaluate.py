'''Test evaluation pipeline entry point (separate from training).

Rebuilds the model from the config, loads a trained checkpoint and
evaluates it on the test set, reporting loss, accuracy, top-k accuracy,
per-class precision/recall/F1 (macro and weighted averages) and the
confusion matrix. Metrics are also saved as JSON.

Usage:
    python evaluate.py --config configs/baseline.yaml --checkpoint checkpoints/best_model.pth

    # Evaluate on a different split / save metrics elsewhere
    python evaluate.py --config configs/baseline.yaml --checkpoint checkpoints/best_model.pth \
        --data-path data/val --output results/val_metrics.json
'''
import argparse
from pathlib import Path

import torch

from evaluation.evaluator import Evaluator
from train.model import build_model_from_config
from utils.config_utils import load_config
from utils.data_utils import build_eval_loader


def load_model_weights(model: torch.nn.Module, checkpoint_path: str, device: str) -> None:
    '''Load trained weights into the model.

    Supports both Trainer checkpoints ({'model_state': ...}) and plain
    state dicts saved with torch.save(model.state_dict(), ...).
    '''
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get('model_state', checkpoint)
    model.load_state_dict(state_dict)

    epoch = checkpoint.get('epoch') if isinstance(checkpoint, dict) else None
    info  = f" (epoch {epoch})" if epoch is not None else ""
    print(f"Checkpoint loaded from {checkpoint_path}{info}")


def main(config_path: str, checkpoint_path: str, data_path: str | None,
         output_path: str | None, batch_size: int | None) -> None:
    cfg = load_config(config_path)

    model_cfg = cfg['model']
    if model_cfg['head']['type'] == 'none':
        raise ValueError(
            "head.type='none' has no classifier to evaluate. "
            "Use extract_features.py for feature-extractor mode."
        )

    # Resolve test set: --data-path overrides dataset.test_path
    data_path = data_path or cfg['dataset'].get('test_path')
    if not data_path:
        raise ValueError(
            "No test set specified. Set 'dataset.test_path' in the config "
            "or pass --data-path."
        )

    device = cfg['training']['device']
    batch_size = batch_size or cfg['training']['batch_size']

    # Model + trained weights
    model = build_model_from_config(model_cfg)
    load_model_weights(model, checkpoint_path, device)

    # Data
    loader = build_eval_loader(data_path, batch_size)
    print(f"Dataset → test: {len(loader.dataset)} images | classes: {loader.dataset.classes}")

    # Evaluation
    eval_cfg  = cfg.get('evaluation') or {}
    topk      = tuple(eval_cfg.get('topk') or (1, 5))
    evaluator = Evaluator(model, device, topk=topk)
    metrics   = evaluator.evaluate(loader, class_names=loader.dataset.classes)

    # Save metrics next to the checkpoint by default
    if output_path is None:
        output_path = Path(checkpoint_path).parent / 'test_metrics.json'
    Evaluator.save_metrics(metrics, output_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DINOv2/DINOv3 test evaluation pipeline')
    parser.add_argument('--config',     type=str, required=True, help='Path to the YAML configuration file')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to the trained checkpoint .pth')
    parser.add_argument('--data-path',  type=str, default=None,  help='ImageFolder to evaluate (overrides dataset.test_path)')
    parser.add_argument('--output',     type=str, default=None,  help='Where to save the metrics JSON (default: alongside the checkpoint)')
    parser.add_argument('--batch-size', type=int, default=None,  help='Batch size (default: training.batch_size)')
    args = parser.parse_args()

    main(args.config, args.checkpoint, args.data_path, args.output, args.batch_size)
