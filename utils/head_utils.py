import torch.nn as nn


def linear_head(embed_dim: int, num_classes: int) -> nn.Module:
    '''Linear classification head: a single Linear layer.

    Args:
        embed_dim:   Embedding dimension of the backbone.
        num_classes: Number of output classes.
    '''
    return nn.Linear(embed_dim, num_classes)


def mlp_head(
    embed_dim: int,
    num_classes: int,
    hidden_dim: int,
    dropout: float = 0.0,
) -> nn.Module:
    '''MLP classification head with one hidden layer.

    Architecture: Linear → GELU → Dropout → Linear

    Args:
        embed_dim:   Embedding dimension of the backbone.
        num_classes: Number of output classes.
        hidden_dim:  Dimension of the hidden layer.
        dropout:     Dropout rate (0.0 disables dropout).
    '''
    return nn.Sequential(
        nn.Linear(embed_dim, hidden_dim),
        nn.GELU(),
        nn.Dropout(p=dropout),
        nn.Linear(hidden_dim, num_classes),
    )


def build_head(head_type: str, embed_dim: int, num_classes: int | None, **kwargs) -> nn.Module:
    '''Factory function to build a classification head.

    Args:
        head_type:   Head type. Options: 'linear', 'mlp', 'none'.
                     'none' returns an Identity: the model outputs raw
                     backbone features (feature extractor mode).
        embed_dim:   Embedding dimension of the backbone.
        num_classes: Number of output classes (ignored when type='none').
        **kwargs:    Additional parameters depending on the type:
                       - mlp: hidden_dim (int), dropout (float, optional).
    '''
    if head_type == 'none':
        return nn.Identity()

    if num_classes is None:
        raise ValueError(f"num_classes is required for head type '{head_type}'.")

    if head_type == 'linear':
        return linear_head(embed_dim, num_classes)

    if head_type == 'mlp':
        hidden_dim = kwargs.get('hidden_dim', embed_dim // 2)
        dropout    = kwargs.get('dropout', 0.0)
        return mlp_head(embed_dim, num_classes, hidden_dim, dropout)

    raise ValueError(f"Head type '{head_type}' is not supported. Options: 'linear', 'mlp', 'none'.")
