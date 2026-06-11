from pathlib import Path

import yaml

from utils.model_utils import BACKBONE_FAMILIES


# Required fields per section
_REQUIRED_FIELDS = {
    'training':      ['epochs', 'batch_size', 'learning_rate', 'optimizer', 'loss_function', 'device'],
    'model':         ['family', 'variant', 'weights_path', 'freeze_ratio'],
    'model.head':    ['type'],
    'dataset':       ['train_path', 'val_path'],
}

_VALID_OPTIMIZERS   = ('adam', 'adamw', 'sgd')
_VALID_LOSSES       = ('cross_entropy',)
_VALID_HEAD_TYPES   = ('linear', 'mlp', 'none')
_VALID_DEVICES      = ('cuda', 'cpu')


def load_config(path: str) -> dict:
    '''Load a YAML configuration file and validate required fields.

    Args:
        path: Path to the .yaml configuration file.

    Returns:
        Dictionary with the validated configuration.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError:        If required fields are missing or values are invalid.
    '''
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with open(path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    _validate_required_fields(cfg)
    _validate_values(cfg)

    return cfg


# ------------------------------------------------------------------
# Validation helpers
# ------------------------------------------------------------------

def _validate_required_fields(cfg: dict) -> None:
    errors = []

    for section, fields in _REQUIRED_FIELDS.items():
        if '.' in section:
            parent, child = section.split('.', 1)
            block = (cfg.get(parent) or {}).get(child) or {}
        else:
            block = cfg.get(section) or {}

        for field in fields:
            if block.get(field) is None:
                errors.append(f"  [{section}] '{field}' is required but not set.")

    if errors:
        raise ValueError("Incomplete configuration:\n" + "\n".join(errors))


def _validate_values(cfg: dict) -> None:
    training = cfg['training']
    model    = cfg['model']
    head     = model['head']
    errors   = []

    if training['optimizer'] not in _VALID_OPTIMIZERS:
        errors.append(f"  'optimizer' must be one of {_VALID_OPTIMIZERS}.")

    if training['loss_function'] not in _VALID_LOSSES:
        errors.append(f"  'loss_function' must be one of {_VALID_LOSSES}.")

    if training['device'] not in _VALID_DEVICES:
        errors.append(f"  'device' must be one of {_VALID_DEVICES}.")

    family = model['family']
    if family not in BACKBONE_FAMILIES:
        errors.append(f"  'family' must be one of {tuple(BACKBONE_FAMILIES)}.")
    else:
        valid_variants = tuple(BACKBONE_FAMILIES[family]['variants'])
        if model['variant'] not in valid_variants:
            errors.append(f"  'variant' must be one of {valid_variants} for family '{family}'.")

    if not (0.0 <= float(model['freeze_ratio']) <= 1.0):
        errors.append("  'freeze_ratio' must be between 0.0 and 1.0.")

    if head['type'] not in _VALID_HEAD_TYPES:
        errors.append(f"  'head.type' must be one of {_VALID_HEAD_TYPES}.")

    if head['type'] != 'none' and model.get('num_classes') is None:
        errors.append("  'num_classes' is required when head.type is 'linear' or 'mlp'.")

    if head['type'] == 'mlp' and head.get('hidden_dim') is None:
        errors.append("  'head.hidden_dim' is required when type='mlp'.")

    if errors:
        raise ValueError("Invalid configuration values:\n" + "\n".join(errors))
