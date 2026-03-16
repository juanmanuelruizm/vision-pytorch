import torch
import torch.nn as nn

from utils.model_utils import load_backbone, apply_freeze
from utils.head_utils import build_head


class Model(nn.Module):
    '''DINOv2 model with a configurable classification head.

    Combines a DINOv2 ViT backbone (loaded from Meta's local weights)
    with a linear or MLP classification head.

    Args:
        variant:      ViT backbone variant. Options: 'vit_small', 'vit_base',
                      'vit_large', 'vit_giant'.
        weights_path: Path to the .pth file with Meta-provided backbone weights.
        num_classes:  Number of output classes.
        head_type:    Classification head type. Options: 'linear', 'mlp'.
                      Defaults to 'linear'.
        freeze_ratio: Fraction of the backbone to freeze (0.0 to 1.0).
                      0.0 = nothing frozen, 1.0 = fully frozen. Defaults to 0.0.
        **kwargs:     Additional parameters for the head:
                        - mlp: hidden_dim (int), dropout (float).
    '''

    def __init__(
        self,
        variant: str,
        weights_path: str,
        num_classes: int,
        head_type: str = 'linear',
        freeze_ratio: float = 0.0,
        **kwargs,
    ):
        super().__init__()

        self.backbone, embed_dim = load_backbone(variant, weights_path)
        apply_freeze(self.backbone, freeze_ratio)
        self.head = build_head(head_type, embed_dim, num_classes, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''Forward pass: backbone → classification head.

        DINOv2 returns the CLS token as the global image representation
        (shape: [B, embed_dim]), which is passed directly to the head.
        '''
        features = self.backbone(x)
        return self.head(features)
