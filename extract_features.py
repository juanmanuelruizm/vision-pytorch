'''Feature extraction pipeline: use the DINO backbone as a raw feature extractor.

Runs the backbone (no classification head) over an ImageFolder dataset and
saves the embeddings to a .pt file with:
    {
        'features': FloatTensor [N, embed_dim],
        'labels':   LongTensor  [N],
        'paths':    list[str]   (image file paths),
        'classes':  list[str]   (class names in label order),
    }

The saved features can be used for k-NN / linear probes, clustering,
retrieval, or training lightweight heads without re-running the backbone.

Usage:
    python extract_features.py --config configs/baseline.yaml \
        --data-path data/train --output features/train_features.pt

    # From a fine-tuned checkpoint instead of the raw Meta weights
    python extract_features.py --config configs/baseline.yaml \
        --data-path data/test --output features/test_features.pt \
        --checkpoint checkpoints/best_model.pth
'''
import argparse
from pathlib import Path

import torch

from train.model import build_model_from_config
from utils.config_utils import load_config
from utils.data_utils import build_eval_loader


def main(config_path: str, data_path: str, output_path: str,
         checkpoint_path: str | None, batch_size: int | None) -> None:
    cfg    = load_config(config_path)
    device = cfg['training']['device']
    batch_size = batch_size or cfg['training']['batch_size']

    # Build the model; only the backbone is used here, so any head config works.
    model = build_model_from_config(cfg['model'])

    # Optionally load fine-tuned weights (Trainer checkpoint or plain state dict)
    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        state_dict = checkpoint.get('model_state', checkpoint)
        model.load_state_dict(state_dict, strict=False)
        print(f"Checkpoint loaded from {checkpoint_path}")

    model = model.to(device)
    model.eval()

    loader = build_eval_loader(data_path, batch_size)
    print(f"Dataset → {data_path}: {len(loader.dataset)} images | classes: {loader.dataset.classes}")

    all_features = []
    all_labels   = []
    with torch.no_grad():
        for i, (x, y) in enumerate(loader, start=1):
            features = model.extract_features(x.to(device))
            all_features.append(features.cpu())
            all_labels.append(y)
            if i % 50 == 0 or i == len(loader):
                print(f"  Batch {i}/{len(loader)} processed")

    features = torch.cat(all_features)
    labels   = torch.cat(all_labels)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'features': features,
        'labels':   labels,
        'paths':    [path for path, _ in loader.dataset.samples],
        'classes':  loader.dataset.classes,
    }, output_path)

    print(f"\nFeatures saved to: {output_path} | shape: {tuple(features.shape)}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DINOv2/DINOv3 feature extraction pipeline')
    parser.add_argument('--config',     type=str, required=True, help='Path to the YAML configuration file')
    parser.add_argument('--data-path',  type=str, required=True, help='ImageFolder dataset to extract features from')
    parser.add_argument('--output',     type=str, required=True, help='Output .pt file for the extracted features')
    parser.add_argument('--checkpoint', type=str, default=None,  help='Optional fine-tuned checkpoint (default: raw Meta weights)')
    parser.add_argument('--batch-size', type=int, default=None,  help='Batch size (default: training.batch_size)')
    args = parser.parse_args()

    main(args.config, args.data_path, args.output, args.checkpoint, args.batch_size)
