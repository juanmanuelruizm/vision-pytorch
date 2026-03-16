import torch


DINOV2_VARIANTS = {
    'vit_small': ('dinov2_vits14', 384),
    'vit_base':  ('dinov2_vitb14', 768),
    'vit_large': ('dinov2_vitl14', 1024),
    'vit_giant': ('dinov2_vitg14', 1536),
}


def load_backbone(variant: str, weights_path: str) -> tuple[torch.nn.Module, int]:
    '''Instantiate a DINOv2 ViT architecture and load weights from a local .pth file.

    Args:
        variant:      Model variant. Options: 'vit_small', 'vit_base', 'vit_large', 'vit_giant'.
        weights_path: Path to the .pth file with Meta-provided weights.

    Returns:
        backbone:  ViT model with loaded weights.
        embed_dim: Embedding dimension of the backbone.
    '''
    if variant not in DINOV2_VARIANTS:
        raise ValueError(
            f"Variant '{variant}' is not supported. "
            f"Available options: {list(DINOV2_VARIANTS.keys())}"
        )

    hub_name, embed_dim = DINOV2_VARIANTS[variant]

    # Instantiate the architecture without pretrained weights
    backbone = torch.hub.load('facebookresearch/dinov2', hub_name, pretrained=False)

    # Load local weights from the Meta-provided .pth file
    state_dict = torch.load(weights_path, map_location='cpu')

    # Meta's .pth files may include extra keys (e.g. 'head'). Filter them out.
    model_keys = set(backbone.state_dict().keys())
    filtered_state_dict = {k: v for k, v in state_dict.items() if k in model_keys}

    missing_keys = model_keys - set(filtered_state_dict.keys())
    if missing_keys:
        print(f"[Warning] Keys not found in .pth file: {missing_keys}")

    backbone.load_state_dict(filtered_state_dict, strict=False)

    return backbone, embed_dim


def apply_freeze(backbone: torch.nn.Module, freeze_ratio: float) -> None:
    '''Freeze a fraction of the backbone layers according to freeze_ratio.

    Freezing is applied in depth order:
      - freeze_ratio = 0.0 → nothing frozen (full fine-tuning).
      - freeze_ratio = 1.0 → entire backbone frozen (linear probe).
      - 0 < freeze_ratio < 1 → embedding layers + first floor(ratio * n_blocks) transformer blocks.

    Args:
        backbone:     DINOv2 backbone (ViT instance).
        freeze_ratio: Fraction to freeze, between 0.0 and 1.0.
    '''
    if not (0.0 <= freeze_ratio <= 1.0):
        raise ValueError("freeze_ratio must be between 0.0 and 1.0.")

    if freeze_ratio == 0.0:
        return

    if freeze_ratio == 1.0:
        for param in backbone.parameters():
            param.requires_grad = False
        return

    # Always freeze the embedding layers (earliest part of the network)
    _freeze_embeddings(backbone)

    # Freeze the first N transformer blocks according to the ratio
    n_blocks = len(backbone.blocks)
    n_freeze = int(freeze_ratio * n_blocks)

    for block in backbone.blocks[:n_freeze]:
        for param in block.parameters():
            param.requires_grad = False

    trainable = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in backbone.parameters())
    print(
        f"[Freeze] {n_freeze}/{n_blocks} blocks frozen + embeddings. "
        f"Trainable backbone params: {trainable:,} / {total:,}"
    )


def _freeze_embeddings(backbone: torch.nn.Module) -> None:
    '''Freeze the embedding layers of the backbone (patch_embed, cls_token, pos_embed).'''
    embedding_keys = ('patch_embed', 'cls_token', 'pos_embed')
    for name, param in backbone.named_parameters():
        if any(key in name for key in embedding_keys):
            param.requires_grad = False
