from argparse import Namespace
from pathlib import Path
from typing import Dict, List, Tuple

from PIL import Image
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms as T


def get_inbreast_paths(config: Namespace) -> Dict[str, List[Path]]:
    """
    Returns paths for the prepared INBreast dataset.

    Expected dataset structure:
        datasets/INBreast/
        ├── train/
        │   └── healthy/
        └── test/
            ├── healthy/
            ├── mass/
            └── masks/
    """

    root = Path(config.datasets_dir) / "INBreast"

    paths = {
        "train_healthy": sorted((root / "train" / "healthy").glob("*.png")),
        "test_healthy": sorted((root / "test" / "healthy").glob("*.png")),
        "test_mass": sorted((root / "test" / "mass").glob("*.png")),
        "test_masks": sorted((root / "test" / "masks").glob("*_mask.png")),
    }

    return paths


class INBreastHealthyDataset(Dataset):
    """
    Dataset used for training.

    Only healthy mammography images are used during training because the model
    learns to reconstruct normal breast tissue. Anomalies are introduced later
    through the reconstruction error during evaluation.
    """

    def __init__(self, image_paths: List[Path], image_size: int = 256, center: bool = True):
        self.image_paths = image_paths
        self.center = center

        self.transform = T.Compose([
            T.Resize((image_size, image_size), interpolation=T.InterpolationMode.LANCZOS),
            T.ToTensor(),
        ])

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> Tensor:
        image = Image.open(self.image_paths[index]).convert("L")
        image = self.transform(image)

        if self.center:
            image = (image - 0.5) * 2.0

        return image


class INBreastEvaluationDataset(Dataset):
    """
    Dataset used for evaluation.

    It contains both healthy and mass images. Healthy samples have empty masks,
    while mass samples have binary ground-truth masks.
    """

    def __init__(
        self,
        healthy_paths: List[Path],
        mass_paths: List[Path],
        mask_paths: List[Path],
        image_size: int = 256,
        center: bool = True,
    ):
        self.center = center

        self.images = mass_paths + healthy_paths
        self.labels = [1] * len(mass_paths) + [0] * len(healthy_paths)

        mask_by_stem = {
            mask_path.name.replace("_mask.png", ""): mask_path
            for mask_path in mask_paths
        }

        self.mask_paths = []
        for image_path in mass_paths:
            case_id = image_path.stem
            if case_id not in mask_by_stem:
                raise FileNotFoundError(f"Missing mask for mass image: {image_path.name}")
            self.mask_paths.append(mask_by_stem[case_id])

        self.image_transform = T.Compose([
            T.Resize((image_size, image_size), interpolation=T.InterpolationMode.LANCZOS),
            T.ToTensor(),
        ])

        self.mask_transform = T.Compose([
            T.Resize((image_size, image_size), interpolation=T.InterpolationMode.NEAREST),
            T.ToTensor(),
        ])

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor, int, str]:
        image_path = self.images[index]
        label = self.labels[index]

        image = Image.open(image_path).convert("L")
        image = self.image_transform(image)

        if self.center:
            image = (image - 0.5) * 2.0

        if label == 1:
            mask_index = index
            mask = Image.open(self.mask_paths[mask_index]).convert("L")
            mask = self.mask_transform(mask)
            mask = (mask > 0.5).float()
        else:
            mask = torch.zeros_like(image)

        return image, mask, label, str(image_path)


def get_datasets_inbreast(config: Namespace, train: bool = True):
    """
    Creates INBreast datasets.

    If train=True, returns:
        train_dataset, validation_dataset

    If train=False, returns:
        evaluation_dataset
    """

    paths = get_inbreast_paths(config)

    if train:
        full_dataset = INBreastHealthyDataset(
            image_paths=paths["train_healthy"],
            image_size=config.image_size,
            center=config.center,
        )

        train_size = int(len(full_dataset) * config.normal_split)
        val_size = len(full_dataset) - train_size

        generator = torch.Generator().manual_seed(config.seed)
        train_dataset, val_dataset = random_split(
            full_dataset,
            [train_size, val_size],
            generator=generator,
        )

        return train_dataset, val_dataset

    evaluation_dataset = INBreastEvaluationDataset(
        healthy_paths=paths["test_healthy"],
        mass_paths=paths["test_mass"],
        mask_paths=paths["test_masks"],
        image_size=config.image_size,
        center=config.center,
    )

    return evaluation_dataset


def get_dataloaders_inbreast(config: Namespace, train: bool = True):
    """
    Creates PyTorch DataLoader objects for INBreast.
    """

    if train:
        train_dataset, val_dataset = get_datasets_inbreast(config, train=True)

        train_loader = DataLoader(
            train_dataset,
            batch_size=config.train_batch_size,
            shuffle=True,
            num_workers=config.dataloader_num_workers,
            pin_memory=True,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=config.validation_batch_size,
            shuffle=False,
            num_workers=config.dataloader_num_workers,
            pin_memory=True,
        )

        return train_loader, val_loader

    evaluation_dataset = get_datasets_inbreast(config, train=False)

    evaluation_loader = DataLoader(
        evaluation_dataset,
        batch_size=config.validation_batch_size,
        shuffle=False,
        num_workers=config.dataloader_num_workers,
        pin_memory=True,
    )

    return evaluation_loader