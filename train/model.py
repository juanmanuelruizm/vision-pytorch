import torch
import torch.nn as nn

from utils.model_utils import load_backbone, apply_freeze
from utils.head_utils import build_head


class Model(nn.Module):
    '''DINO model with a configurable classification head.

    Combines a DINOv2 or DINOv3 ViT backbone (loaded from Meta's local
    weights) with an optional classification head. With head_type='none'
    the model acts as a pure feature extractor: forward() returns the
    backbone embeddings directly.

    Args:
        family:       Backbone family. Options: 'dinov2', 'dinov3'.
        variant:      ViT backbone variant within the family
                      (e.g. 'vit_small', 'vit_base', 'vit_large', ...).
        weights_path: Path to the .pth file with Meta-provided backbone weights.
        num_classes:  Number of output classes. Not required when head_type='none'.
        head_type:    Classification head type. Options: 'linear', 'mlp', 'none'.
                      Defaults to 'linear'.
        freeze_ratio: Fraction of the backbone to freeze (0.0 to 1.0).
                      0.0 = nothing frozen, 1.0 = fully frozen. Defaults to 0.0.
        **kwargs:     Additional parameters for the head:
                        - mlp: hidden_dim (int), dropout (float).
    '''

    def __init__(
        self,
        family: str,
        variant: str,
        weights_path: str,
        num_classes: int | None = None,
        head_type: str = 'linear',
        freeze_ratio: float = 0.0,
        **kwargs,
    ):
        super().__init__()

        self.backbone, self.embed_dim = load_backbone(family, variant, weights_path)
        apply_freeze(self.backbone, freeze_ratio)
        self.head = build_head(head_type, self.embed_dim, num_classes, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''Forward pass: backbone → classification head.

        DINO backbones return the CLS token as the global image
        representation (shape: [B, embed_dim]), which is passed to the
        head — or returned as-is when head_type='none'.
        '''
        features = self.backbone(x)
        return self.head(features)

    @torch.no_grad()
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        '''Return raw backbone embeddings (shape: [B, embed_dim]), skipping the head.'''
        return self.backbone(x)


def build_model_from_config(model_cfg: dict) -> Model:
    '''Build a Model from the 'model' section of a validated config.

    Shared by the train, evaluation and feature-extraction pipelines so
    the architecture is always reconstructed identically.
    '''
    head_cfg  = model_cfg['head']
    head_type = head_cfg['type']

    head_kwargs = {}
    if head_type == 'mlp':
        head_kwargs['hidden_dim'] = head_cfg['hidden_dim']
        head_kwargs['dropout']    = head_cfg.get('dropout') or 0.0

    return Model(
        family       = model_cfg.get('family', 'dinov2'),
        variant      = model_cfg['variant'],
        weights_path = model_cfg['weights_path'],
        num_classes  = model_cfg.get('num_classes'),
        head_type    = head_type,
        freeze_ratio = float(model_cfg['freeze_ratio']),
        **head_kwargs,
    )
