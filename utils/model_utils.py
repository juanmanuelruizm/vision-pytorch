import torch


# ------------------------------------------------------------------
# Supported backbone families and variants.
#
# Each family maps to the torch.hub repo that provides the architecture
# and the set of supported variants: variant -> (hub_name, embed_dim).
#
# Weights are NEVER downloaded from the hub: the architecture is
# instantiated without pretrained weights and the local .pth file
# (Meta-provided) is loaded on top.
# ------------------------------------------------------------------

BACKBONE_FAMILIES = {
    'dinov2': {
        'hub_repo':   'facebookresearch/dinov2',
        'patch_size': 14,
        'variants': {
            'vit_small': ('dinov2_vits14', 384),
            'vit_base':  ('dinov2_vitb14', 768),
            'vit_large': ('dinov2_vitl14', 1024),
            'vit_giant': ('dinov2_vitg14', 1536),
        },
    },
    'dinov3': {
        'hub_repo':   'facebookresearch/dinov3',
        'patch_size': 16,
        'variants': {
            'vit_small':      ('dinov3_vits16',     384),
            'vit_small_plus': ('dinov3_vits16plus', 384),
            'vit_base':       ('dinov3_vitb16',     768),
            'vit_large':      ('dinov3_vitl16',     1024),
            'vit_huge_plus':  ('dinov3_vith16plus', 1280),
            'vit_7b':         ('dinov3_vit7b16',    4096),
        },
    },
}

# Embedding-level parameters (frozen first when freeze_ratio > 0).
# Covers both families: DINOv2 uses pos_embed; DINOv3 uses RoPE and
# storage (register) tokens.
_EMBEDDING_KEYS = (
    'patch_embed', 'cls_token', 'pos_embed',
    'register_tokens', 'storage_tokens', 'mask_token', 'rope_embed',
)


def load_backbone(family: str, variant: str, weights_path: str) -> tuple[torch.nn.Module, int]:
    '''Instantiate a DINO backbone architecture and load weights from a local .pth file.

    Args:
        family:       Backbone family. Options: 'dinov2', 'dinov3'.
        variant:      Model variant within the family (see BACKBONE_FAMILIES).
        weights_path: Path to the .pth file with Meta-provided weights.

    Returns:
        backbone:  Model with loaded weights.
        embed_dim: Embedding dimension of the backbone.
    '''
    if family not in BACKBONE_FAMILIES:
        raise ValueError(
            f"Family '{family}' is not supported. "
            f"Available options: {list(BACKBONE_FAMILIES.keys())}"
        )

    family_cfg = BACKBONE_FAMILIES[family]
    if variant not in family_cfg['variants']:
        raise ValueError(
            f"Variant '{variant}' is not supported for family '{family}'. "
            f"Available options: {list(family_cfg['variants'].keys())}"
        )

    hub_name, embed_dim = family_cfg['variants'][variant]

    # Instantiate the architecture without pretrained weights
    backbone = torch.hub.load(family_cfg['hub_repo'], hub_name, pretrained=False)

    # Load local weights from the Meta-provided .pth file
    state_dict = torch.load(weights_path, map_location='cpu')

    # Some checkpoints wrap the weights (e.g. {'model': ...} or {'teacher': ...})
    for wrapper_key in ('model', 'teacher', 'state_dict'):
        if wrapper_key in state_dict and isinstance(state_dict[wrapper_key], dict):
            state_dict = state_dict[wrapper_key]
            break

    # Meta's .pth files may include extra keys (e.g. 'head'). Filter them out.
    model_keys = set(backbone.state_dict().keys())
    filtered_state_dict = {k: v for k, v in state_dict.items() if k in model_keys}

    missing_keys = model_keys - set(filtered_state_dict.keys())
    if missing_keys:
        print(f"[Warning] Keys not found in .pth file: {missing_keys}")

    backbone.load_state_dict(filtered_state_dict, strict=False)
    print(f"[Backbone] {family}/{variant} ({hub_name}) loaded from {weights_path} | embed_dim: {embed_dim}")

    return backbone, embed_dim


def apply_freeze(backbone: torch.nn.Module, freeze_ratio: float) -> None:
    '''Freeze a fraction of the backbone layers according to freeze_ratio.

    Freezing is applied in depth order:
      - freeze_ratio = 0.0 → nothing frozen (full fine-tuning).
      - freeze_ratio = 1.0 → entire backbone frozen (feature extractor / linear probe).
      - 0 < freeze_ratio < 1 → embedding layers + first floor(ratio * n_blocks) transformer blocks.

    Args:
        backbone:     DINO backbone (ViT instance).
        freeze_ratio: Fraction to freeze, between 0.0 and 1.0.
    '''
    if not (0.0 <= freeze_ratio <= 1.0):
        raise ValueError("freeze_ratio must be between 0.0 and 1.0.")

    if freeze_ratio == 0.0:
        return

    if freeze_ratio == 1.0:
        for param in backbone.parameters():
            param.requires_grad = False
        print("[Freeze] Backbone fully frozen (feature extractor mode).")
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
    '''Freeze the embedding layers of the backbone (patch/cls/pos/register embeddings).'''
    for name, param in backbone.named_parameters():
        if any(key in name for key in _EMBEDDING_KEYS):
            param.requires_grad = False
