"""Dataset utilities for downstream concept bottleneck evaluation."""

from __future__ import annotations

import urllib.request
import zipfile
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image, ImageDraw
from torchvision import datasets


class Cutout:
    """Mask square image regions before backbone preprocessing."""

    def __init__(
        self,
        n_holes: int = 1,
        length: int = 16,
        p: float = 1.0,
        fill_value: int | tuple[int, int, int] = 0,
    ) -> None:
        self.n_holes = int(n_holes)
        self.length = int(length)
        self.p = float(p)
        if isinstance(fill_value, tuple):
            self.fill_value_pil = tuple(int(value) for value in fill_value)
        else:
            self.fill_value_pil = (int(fill_value),) * 3
        self.fill_value_tensor = float(fill_value) if not isinstance(fill_value, tuple) else 0.0

    def __call__(self, image):
        if np.random.rand() > self.p or self.n_holes <= 0 or self.length <= 0:
            return image

        if isinstance(image, Image.Image):
            width, height = image.size
            draw = ImageDraw.Draw(image)
            for _ in range(self.n_holes):
                center_x = np.random.randint(0, width)
                center_y = np.random.randint(0, height)
                half = self.length // 2
                x1, y1 = max(0, center_x - half), max(0, center_y - half)
                x2, y2 = min(width, center_x + half), min(height, center_y + half)
                draw.rectangle([x1, y1, x2, y2], fill=self.fill_value_pil)
            return image

        if torch.is_tensor(image) and image.dim() == 3:
            _, height, width = image.shape
            for _ in range(self.n_holes):
                center_x = np.random.randint(0, width)
                center_y = np.random.randint(0, height)
                half = self.length // 2
                x1, y1 = max(0, center_x - half), max(0, center_y - half)
                x2, y2 = min(width, center_x + half), min(height, center_y + half)
                image[:, y1:y2, x1:x2] = self.fill_value_tensor
        return image


class TinyImageNetDataset(torch.utils.data.Dataset):
    """Tiny ImageNet training or validation split."""

    def __init__(self, root: str | Path, train: bool = True, transform=None, download: bool = False):
        self.root = Path(root)
        self.train = bool(train)
        self.transform = transform
        self.data_dir = self.root / "tiny-imagenet-200"

        if download:
            self._download()
        if not self.data_dir.exists():
            raise RuntimeError("Tiny ImageNet was not found. Set download=True to download it.")

        self.class_to_idx: Dict[str, int] = {}
        with (self.data_dir / "wnids.txt").open("r", encoding="utf-8") as handle:
            for index, line in enumerate(handle):
                self.class_to_idx[line.strip()] = index

        self.samples: list[tuple[str, int]] = []
        if self.train:
            train_dir = self.data_dir / "train"
            for class_name, class_index in self.class_to_idx.items():
                image_dir = train_dir / class_name / "images"
                if image_dir.exists():
                    self.samples.extend(
                        (str(image_path), class_index)
                        for image_path in image_dir.glob("*.JPEG")
                    )
        else:
            val_dir = self.data_dir / "val"
            with (val_dir / "val_annotations.txt").open("r", encoding="utf-8") as handle:
                for line in handle:
                    image_name, class_name, *_ = line.rstrip("\n").split("\t")
                    image_path = val_dir / "images" / image_name
                    if image_path.exists() and class_name in self.class_to_idx:
                        self.samples.append((str(image_path), self.class_to_idx[class_name]))

        self.targets = [label for _, label in self.samples]

    def _download(self) -> None:
        if self.data_dir.exists():
            return
        self.root.mkdir(parents=True, exist_ok=True)
        url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
        archive_path = self.root / "tiny-imagenet-200.zip"
        with urllib.request.urlopen(url) as response, archive_path.open("wb") as output:
            while chunk := response.read(8192):
                output.write(chunk)
        with zipfile.ZipFile(archive_path, "r") as archive:
            archive.extractall(self.root)
        archive_path.unlink(missing_ok=True)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        image_path, label = self.samples[index]
        image = Image.open(image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label


class CUBDatasetWithPreTransform(datasets.ImageFolder):
    """CUB-200 ImageFolder with separate augmentation and backbone transforms."""

    def __init__(
        self,
        root: str | Path,
        pre_transform=None,
        main_transform=None,
        target_transform=None,
    ) -> None:
        super().__init__(root, transform=None, target_transform=target_transform)
        self.pre_transform = pre_transform
        self.main_transform = main_transform

    def __getitem__(self, index: int):
        path, target = self.samples[index]
        with open(path, "rb") as handle:
            image = Image.open(handle).convert("RGB")
        if self.pre_transform is not None:
            image = self.pre_transform(image)
        if self.main_transform is not None:
            image = self.main_transform(image)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return image, target


def get_cifar_dataset(
    dataset_name: str,
    data_dir: str | Path,
    backbone=None,
    augmentation_args: Optional[Dict[str, Any]] = None,
):
    """Build CIFAR-10, CIFAR-100, or Tiny ImageNet train and test datasets."""

    dataset_name = dataset_name.lower()
    if dataset_name == "tiny-imagenet":
        base_size, default_padding, default_cutout = 64, 8, 32
    else:
        base_size, default_padding, default_cutout = 32, 4, 16

    if augmentation_args is None:
        augmentation_args = {
            "use_data_augmentation": True,
            "crop_padding": default_padding,
            "flip_prob": 0.5,
            "cutout_n_holes": 1,
            "cutout_length": default_cutout,
            "cutout_p": 1.0,
        }

    if backbone is not None and getattr(backbone, "preprocess", None) is not None:
        test_transform = backbone.preprocess
    else:
        test_transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    if augmentation_args.get("use_data_augmentation", True):
        augmentations = [
            transforms.RandomCrop(base_size, padding=augmentation_args["crop_padding"]),
            transforms.RandomHorizontalFlip(p=augmentation_args["flip_prob"]),
            Cutout(
                n_holes=augmentation_args["cutout_n_holes"],
                length=augmentation_args["cutout_length"],
                p=augmentation_args.get("cutout_p", 1.0),
                fill_value=0,
            ),
        ]
        if backbone is not None and getattr(backbone, "preprocess", None) is not None:
            train_transform = transforms.Compose(
                augmentations + list(backbone.preprocess.transforms)
            )
        else:
            train_transform = transforms.Compose(
                augmentations
                + [
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225],
                    ),
                ]
            )
    else:
        train_transform = test_transform

    if dataset_name == "cifar10":
        train_dataset = datasets.CIFAR10(
            root=data_dir, train=True, download=True, transform=train_transform
        )
        test_dataset = datasets.CIFAR10(
            root=data_dir, train=False, download=True, transform=test_transform
        )
        num_classes = 10
    elif dataset_name == "cifar100":
        train_dataset = datasets.CIFAR100(
            root=data_dir, train=True, download=True, transform=train_transform
        )
        test_dataset = datasets.CIFAR100(
            root=data_dir, train=False, download=True, transform=test_transform
        )
        num_classes = 100
    elif dataset_name == "tiny-imagenet":
        train_dataset = TinyImageNetDataset(
            root=data_dir, train=True, download=True, transform=train_transform
        )
        test_dataset = TinyImageNetDataset(
            root=data_dir, train=False, download=True, transform=test_transform
        )
        num_classes = 200
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    return train_dataset, test_dataset, num_classes


__all__ = [
    "CUBDatasetWithPreTransform",
    "Cutout",
    "TinyImageNetDataset",
    "get_cifar_dataset",
]
