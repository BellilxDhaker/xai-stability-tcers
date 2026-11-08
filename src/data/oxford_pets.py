from pathlib import Path
from typing import Tuple

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transforms(image_size: int = 224, train: bool = False) -> transforms.Compose:
    operations = [transforms.Resize((image_size, image_size))]
    if train:
        operations.extend([transforms.RandomHorizontalFlip(), transforms.RandomRotation(10)])
    operations.extend([
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return transforms.Compose(operations)


def load_oxford_pets(
    root: Path,
    image_size: int = 224,
    batch_size: int = 32,
    num_workers: int = 2,
    validation_fraction: float = 0.15,
    seed: int = 42,
    target_types: str = "binary-category",
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Load official train/test splits and derive validation only from train.

    ``target_types`` defaults to ``"binary-category"`` (cat vs. dog), matching
    the two-output classifier used throughout the project. Pass ``"category"``
    only when intentionally training the 37-breed task with a matching head.
    """
    train_full = datasets.OxfordIIITPet(
        root=str(root), split="trainval", target_types=target_types,
        transform=build_transforms(image_size, train=True), download=True,
    )
    test = datasets.OxfordIIITPet(
        root=str(root), split="test", target_types=target_types,
        transform=build_transforms(image_size, train=False), download=True,
    )
    generator = torch.Generator().manual_seed(seed)
    val_size = int(len(train_full) * validation_fraction)
    train_size = len(train_full) - val_size
    train, validation = random_split(train_full, [train_size, val_size], generator=generator)
    # Keep augmentation isolated: random_split subsets share their parent dataset.
    train_dataset = datasets.OxfordIIITPet(
        root=str(root), split="trainval", target_types=target_types,
        transform=build_transforms(image_size, train=True), download=False,
    )
    validation_dataset = datasets.OxfordIIITPet(
        root=str(root), split="trainval", target_types=target_types,
        transform=build_transforms(image_size, train=False), download=False,
    )
    train = torch.utils.data.Subset(train_dataset, train.indices)
    validation = torch.utils.data.Subset(validation_dataset, validation.indices)
    kwargs = {"batch_size": batch_size, "num_workers": num_workers,
              "pin_memory": torch.cuda.is_available()}
    return DataLoader(train, shuffle=True, **kwargs), DataLoader(validation, shuffle=False, **kwargs), DataLoader(test, shuffle=False, **kwargs)
