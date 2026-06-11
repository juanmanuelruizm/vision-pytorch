from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# ------------------------------------------------------------------
# Image transforms for DINO backbones.
# 224 is divisible by both patch sizes: 14 (DINOv2) and 16 (DINOv3).
# Normalization with ImageNet statistics (used by both families).
# ------------------------------------------------------------------
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD  = (0.229, 0.224, 0.225)

TRANSFORM_TRAIN = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
])

TRANSFORM_EVAL = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
])


def build_train_loader(path: str, batch_size: int, num_workers: int = 4) -> DataLoader:
    '''Build a shuffled DataLoader with training augmentations over an ImageFolder.'''
    dataset = datasets.ImageFolder(path, transform=TRANSFORM_TRAIN)
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
    )


def build_eval_loader(path: str, batch_size: int, num_workers: int = 4) -> DataLoader:
    '''Build a deterministic DataLoader with eval transforms over an ImageFolder.

    Used for validation, test evaluation and feature extraction.
    '''
    dataset = datasets.ImageFolder(path, transform=TRANSFORM_EVAL)
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
