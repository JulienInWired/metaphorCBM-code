#!/usr/bin/env python3
"""CIFAR and Tiny ImageNet classification with spatial SAE features.

A shared 2D encoder processes each sparse 7x7 ImageSAE activation map before
the final linear classification layer.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple

import torch

from metaphorcbm.config import parse_args_with_config

from .datasets import get_cifar_dataset
from .spatial_encoder_downstream_common import (
    add_common_spatial_encoder_args,
    class_mapping_from_dataset,
    finalize_common_args,
    run_stage,
    str2bool,
)


def build_datasets(
    args: argparse.Namespace,
    backbone: torch.nn.Module,
) -> Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset, int, Dict[str, int]]:
    augmentation_args = {
        "use_data_augmentation": bool(args.use_data_augmentation),
        "crop_padding": int(args.crop_padding),
        "flip_prob": float(args.flip_prob),
        "cutout_n_holes": int(args.cutout_n_holes),
        "cutout_length": int(args.cutout_length),
        "cutout_p": float(args.cutout_p),
    }
    train_dataset, test_dataset, num_classes = get_cifar_dataset(
        args.dataset,
        args.data_dir,
        backbone=backbone,
        augmentation_args=augmentation_args,
    )
    class_to_idx = class_mapping_from_dataset(train_dataset, num_classes)
    test_class_to_idx = class_mapping_from_dataset(test_dataset, num_classes)
    if class_to_idx != test_class_to_idx:
        raise ValueError("train/test class_to_idx differ; check dataset construction.")
    return train_dataset, test_dataset, int(num_classes), class_to_idx


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CIFAR/TinyImageNet SAE spatial-encoder downstream classifier"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="cifar100",
        choices=["cifar10", "cifar100", "tiny-imagenet"],
    )
    parser.add_argument("--data_dir", type=str, default=None)

    parser.add_argument("--use_data_augmentation", type=str2bool, default=False)
    parser.add_argument("--crop_padding", type=int, default=4)
    parser.add_argument("--flip_prob", type=float, default=0.5)
    parser.add_argument("--cutout_n_holes", type=int, default=1)
    parser.add_argument("--cutout_length", type=int, default=16)
    parser.add_argument("--cutout_p", type=float, default=1.0)

    add_common_spatial_encoder_args(
        parser,
        search_val_per_class=50,
        final_val_per_class=50,
        batch_size_extract=256,
        num_workers=4,
    )
    args = parse_args_with_config(parser)
    if not str(args.data_dir or "").strip():
        raise ValueError(
            "data_dir must be provided through --config or --data_dir"
        )
    args.dataset_tag = str(args.dataset).replace("-", "_")
    return finalize_common_args(args)


def main() -> None:
    args = parse_args()
    run_stage(args, build_datasets)
