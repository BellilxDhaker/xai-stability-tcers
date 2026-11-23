from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from torchvision import datasets, transforms

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transforms(image_size: int = 224, train: bool = False) -> transforms.Compose:
    operations = [transforms.Resize((image_size, image_size))]
    if train:
        operations.extend([transforms.RandomHorizontalFlip(), transforms.RandomRotation(10)])
    return transforms.Compose([*operations, transforms.ToTensor()])


def oxford_split(root, split, image_size, augment=False, target_types="binary-category", download=True):
    return datasets.OxfordIIITPet(
        root=str(root), split=split, target_types=target_types,
        transform=build_transforms(image_size, train=augment), download=download,
    )


def split_labels(root: Path, split: str) -> np.ndarray:
    annotation = root / "oxford-iiit-pet" / "annotations" / f"{split}.txt"
    return np.array([int(line.split()[2]) - 1 for line in annotation.read_text().splitlines() if line.strip()])


def stratified_indices(labels, size: int, seed: int) -> list[int]:
    if not 0 < size <= len(labels):
        raise ValueError(f"Requested {size} samples from a split of {len(labels)}.")
    indices = np.arange(len(labels))
    if size == len(labels):
        return indices.tolist()
    selected, _ = train_test_split(indices, train_size=size, stratify=labels, random_state=seed)
    return sorted(selected.tolist())


def train_validation(root, image_size, validation_fraction, seed, target_types="binary-category", augment=True):
    trainval = oxford_split(root, "trainval", image_size, augment, target_types)
    validation = oxford_split(root, "trainval", image_size, target_types=target_types, download=False)
    count = int(len(trainval) * validation_fraction)
    train_ids, val_ids = random_split(
        range(len(trainval)), [len(trainval) - count, count], generator=torch.Generator().manual_seed(seed),
    )
    return Subset(trainval, train_ids.indices), Subset(validation, val_ids.indices)


def make_loader(dataset, batch_size, num_workers, shuffle=False, seed=42):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                      pin_memory=torch.cuda.is_available(), generator=torch.Generator().manual_seed(seed))


def load_oxford_pets(root: Path, image_size: int = 224, batch_size: int = 32, num_workers: int = 2,
                    validation_fraction: float = 0.15, seed: int = 42, target_types: str = "binary-category",
                    limits: tuple[int, int, int] | None = None):
    """Keep the official test split and the existing seeded validation split."""
    train, validation = train_validation(root, image_size, validation_fraction, seed, target_types)
    test = oxford_split(root, "test", image_size, target_types=target_types)
    if limits:
        train, validation = (Subset(split, range(min(len(split), limit)))
                             for split, limit in zip((train, validation), limits[:2]))
        test = Subset(test, stratified_indices(split_labels(root, "test"), limits[2], seed))
    return (make_loader(train, batch_size, num_workers, shuffle=True, seed=seed),
            make_loader(validation, batch_size, num_workers, seed=seed),
            make_loader(test, batch_size, num_workers, seed=seed))


class IndexedSubset(Dataset):
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, position):
        index = self.indices[position]
        image, label = self.dataset[index]
        return image, label, index


def evaluation_loader(config):
    test = oxford_split(config.data_root, "test", config.image_size)
    indices = stratified_indices(split_labels(config.data_root, "test"), config.test_samples, config.seed)
    return make_loader(IndexedSubset(test, indices), config.evaluation_batch_size, config.num_workers)


def dataset_tables(config):
    import pandas as pd

    train, validation = train_validation(config.data_root, config.image_size, config.validation_fraction, config.seed)
    training_labels = split_labels(config.data_root, "trainval")
    test_labels = split_labels(config.data_root, "test")
    train_ids, validation_ids = train.indices, validation.indices
    if config.quick:
        train_ids = train_ids[:config.quick_train_samples]
        validation_ids = validation_ids[:config.quick_validation_samples]
    selections = {
        "train": training_labels[train_ids],
        "validation": training_labels[validation_ids],
        "test": test_labels,
        "evaluation_cohort": test_labels[stratified_indices(test_labels, config.test_samples, config.seed)],
    }
    return pd.DataFrame(
        {"split": split, "n": len(labels), "cat": (labels == 0).sum(), "dog": (labels == 1).sum()}
        for split, labels in selections.items()
    )
