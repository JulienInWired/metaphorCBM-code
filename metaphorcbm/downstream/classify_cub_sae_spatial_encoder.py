#!/usr/bin/env python3
"""CUB-200 classification with spatial SAE features.

A shared 2D encoder processes each sparse 7x7 ImageSAE activation map before
the final linear classification layer.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple

import torch
import torchvision.transforms as transforms
from torchvision import datasets

from metaphorcbm.config import parse_args_with_config

from .datasets import CUBDatasetWithPreTransform
from .spatial_encoder_downstream_common import (
    add_common_spatial_encoder_args,
    class_mapping_from_dataset,
    finalize_common_args,
    run_stage,
    str2bool,
)


def build_cub_transforms(
    backbone: torch.nn.Module,
    *,
    is_training: bool,
    apply_train_augmentation: bool,
) -> Dict[str, transforms.Compose]:
    pre_transforms = [transforms.Resize((224, 224))]
    if is_training and apply_train_augmentation:
        pre_transforms.extend(
            [
                transforms.RandomRotation(10),
                transforms.RandomHorizontalFlip(p=0.2),
            ]
        )

    if hasattr(backbone, "preprocess") and backbone.preprocess is not None:
        main_transform = backbone.preprocess
    else:
        main_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )
    return {
        "pre_transform": transforms.Compose(pre_transforms),
        "main_transform": main_transform,
    }


def build_datasets(
    args: argparse.Namespace,
    backbone: torch.nn.Module,
) -> Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset, int, Dict[str, int]]:
    train_root = Path(args.data_dir) / "cub200" / "train"
    test_root = Path(args.data_dir) / "cub200" / "test"
    if not train_root.is_dir():
        raise FileNotFoundError(f"CUB-200 train directory not found: {train_root}")
    if not test_root.is_dir():
        raise FileNotFoundError(f"CUB-200 test directory not found: {test_root}")

    train_transforms = build_cub_transforms(
        backbone,
        is_training=True,
        apply_train_augmentation=bool(getattr(args, "_cub_apply_train_augmentation", False)),
    )
    test_transforms = build_cub_transforms(
        backbone,
        is_training=False,
        apply_train_augmentation=False,
    )
    train_dataset = CUBDatasetWithPreTransform(
        root=train_root,
        pre_transform=train_transforms["pre_transform"],
        main_transform=train_transforms["main_transform"],
    )
    test_dataset = CUBDatasetWithPreTransform(
        root=test_root,
        pre_transform=test_transforms["pre_transform"],
        main_transform=test_transforms["main_transform"],
    )
    if train_dataset.class_to_idx != test_dataset.class_to_idx:
        raise ValueError("CUB train/test class_to_idx differ.")
    num_classes = len(train_dataset.classes)
    class_to_idx = class_mapping_from_dataset(train_dataset, num_classes)
    print(
        "CUB train augmentation: "
        + ("enabled" if getattr(args, "_cub_apply_train_augmentation", False) else "disabled")
    )
    return train_dataset, test_dataset, int(num_classes), class_to_idx


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CUB-200 SAE spatial-encoder downstream classifier")
    parser.add_argument("--data_dir", type=str, default=None)

    add_common_spatial_encoder_args(
        parser,
        search_val_per_class=3,
        final_val_per_class=3,
        batch_size_extract=128,
        num_workers=4,
    )
    parser.add_argument(
        "--finetuned_model_path",
        type=str,
        default="",
        help="Optional finetuned CLIP visual-backbone checkpoint used during feature extraction.",
    )
    parser.add_argument(
        "--train_feature_source",
        type=str,
        default="cache",
        choices=["cache", "online"],
        help="Final training input: cached feature shards or online image batches.",
    )
    parser.add_argument(
        "--online_train_augmentation",
        type=str2bool,
        default=False,
        help="Apply training augmentation to online image batches.",
    )
    parser.add_argument(
        "--cub_final_protocol",
        type=str,
        default="holdout_best",
        choices=["holdout_best", "full_train_last"],
        help=(
            "holdout_best selects a checkpoint using validation data; full_train_last "
            "trains on all training images and evaluates the last checkpoint on the test set."
        ),
    )
    args = parse_args_with_config(parser)
    if not str(args.data_dir or "").strip():
        raise ValueError(
            "data_dir must be provided through --config or --data_dir"
        )
    args.dataset_tag = "cub200"
    return finalize_common_args(args)


def main() -> None:
    args = parse_args()
    run_stage(args, build_datasets)
