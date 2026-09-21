#!/usr/bin/env python3
"""Shared spatial-encoder evaluation utilities for small image datasets."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch


from .classify_imagenet_sae_spatial_encoder import (
    FeatureShardStream,
    OnlineFeatureStream,
    build_classifier,
    build_optimizer,
    create_image_sae,
    create_resnet_backbone,
    evaluate_stream,
    extract_split_to_shards,
    indices_sha256,
    load_image_sae_weights,
    materialize_train_subset,
    parse_bool_grid_or_single,
    parse_float_grid_or_single,
    parse_int_grid_or_single,
    parse_seed_grid_or_single,
    read_json,
    set_seed,
    str2bool,
    torch_load,
    train_with_validation,
    verify_split_hashes,
    verify_selection_splits,
    write_json,
)


DatasetFactory = Callable[
    [argparse.Namespace, torch.nn.Module],
    Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset, int, Dict[str, int]],
]


def class_mapping_from_dataset(dataset: Any, num_classes: int) -> Dict[str, int]:
    class_to_idx = getattr(dataset, "class_to_idx", None)
    if isinstance(class_to_idx, dict) and class_to_idx:
        return {str(k): int(v) for k, v in class_to_idx.items()}
    classes = getattr(dataset, "classes", None)
    if classes is not None and len(classes) == num_classes:
        return {str(cls): idx for idx, cls in enumerate(classes)}
    return {str(idx): idx for idx in range(int(num_classes))}


def add_common_spatial_encoder_args(
    parser: argparse.ArgumentParser,
    *,
    search_val_per_class: int,
    final_val_per_class: int,
    batch_size_extract: int,
    num_workers: int,
) -> None:
    parser.add_argument("--stage", type=str, default="prepare", choices=["prepare", "select", "final", "all"])
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--feature_dir", type=str, default=None)
    parser.add_argument("--selection_json", type=str, default=None)
    parser.add_argument("--split_file", type=str, default=None)

    parser.add_argument("--sae_checkpoint", type=str, default=None)
    parser.add_argument("--sae_hidden_dim", type=int, default=8192)
    parser.add_argument("--sae_k_sparse", type=int, default=64)

    parser.add_argument("--use_clip", type=str2bool, default=True)
    parser.add_argument("--clip_variant", type=str, default="RN50")
    parser.add_argument("--clip_pretrained", type=str, default="openai")

    parser.add_argument("--batch_size_extract", type=int, default=batch_size_extract)
    parser.add_argument("--batch_size_head", type=int, default=2048)
    parser.add_argument("--shard_size", type=int, default=4096)
    parser.add_argument("--feature_dtype", type=str, default="float32", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--num_workers", type=int, default=num_workers)
    parser.add_argument("--pin_memory", type=str2bool, default=True)
    parser.add_argument("--overwrite_features", type=str2bool, default=False)
    parser.add_argument("--subset_shard_size", type=int, default=4096)
    parser.add_argument("--overwrite_subsets", type=str2bool, default=False)

    parser.add_argument("--extract_amp", type=str2bool, default=False)
    parser.add_argument("--extract_amp_dtype", type=str, default="float16", choices=["float16", "bfloat16"])
    parser.add_argument("--spatial_topk", type=int, default=64)
    parser.add_argument("--spatial_dim", type=int, default=49, help=argparse.SUPPRESS)
    parser.add_argument("--classifier_mode", type=str, default="spatial", choices=["spatial", "mc"])
    parser.add_argument("--encoder_hidden_channels", type=int, default=8)
    parser.add_argument("--encoder_out_channels", type=int, default=4)
    parser.add_argument("--encoder_kernel_size", type=int, default=3)
    parser.add_argument("--encoder_microbatch_size", type=int, default=1024)
    parser.add_argument("--active_map_chunk_size", type=int, default=65536)
    parser.add_argument("--mc_channel_epsilon", type=float, default=1e-4)
    parser.add_argument("--mc_sqrt_count", type=str2bool, default=True)
    parser.add_argument("--mc_blocknorm", type=str, default="stat", choices=["none", "stat"])

    parser.add_argument(
        "--search_train_per_class",
        type=int,
        default=0,
        help="Internal search train samples per class. 0 means all remaining samples after val splits.",
    )
    parser.add_argument("--search_val_per_class", type=int, default=search_val_per_class)
    parser.add_argument("--final_val_per_class", type=int, default=final_val_per_class)
    parser.add_argument("--overwrite_split", type=str2bool, default=False)
    parser.add_argument("--split_seed", type=int, default=None)
    parser.add_argument("--expected_split_hashes", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seed_grid", type=str, default=None)

    parser.add_argument("--search_epochs", type=int, default=50)
    parser.add_argument("--final_epochs", type=int, default=100)
    parser.add_argument("--final_scheduler_t_max", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--optimizer", type=str, default="sgd", choices=["adamw", "adam", "sgd"])
    parser.add_argument("--sgd_momentum", type=float, default=0.9)
    parser.add_argument("--sgd_nesterov", type=str2bool, default=False)
    parser.add_argument("--sgd_momentum_grid", type=str, default=None)
    parser.add_argument("--sgd_nesterov_grid", type=str, default=None)
    parser.add_argument("--lr_grid", type=str, default=None)
    parser.add_argument("--weight_decay_grid", type=str, default=None)
    parser.add_argument("--batch_size_head_grid", type=str, default=None)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--save_select_checkpoints", type=str2bool, default=False)
    parser.add_argument("--device", type=str, default="cuda")


def finalize_common_args(args: argparse.Namespace) -> argparse.Namespace:
    missing_paths = [
        name
        for name in ("output_dir", "sae_checkpoint")
        if not str(getattr(args, name, "") or "").strip()
    ]
    if missing_paths:
        raise ValueError(
            "The following paths must be provided through --config or command-line "
            "arguments: " + ", ".join(missing_paths)
        )
    if args.split_seed is None:
        args.split_seed = int(args.seed)
    if not hasattr(args, "train_feature_source"):
        args.train_feature_source = "cache"
    if not hasattr(args, "online_train_augmentation"):
        args.online_train_augmentation = False
    if not hasattr(args, "cub_final_protocol"):
        args.cub_final_protocol = "holdout_best"
    if args.epochs is not None:
        args.search_epochs = int(args.epochs)
        args.final_epochs = int(args.epochs)
    if args.search_epochs <= 0 or args.final_epochs <= 0 or args.final_scheduler_t_max <= 0:
        raise ValueError("Epoch counts and --final_scheduler_t_max must be positive.")
    if args.weight_decay < 0.0:
        raise ValueError("--weight_decay must be non-negative.")
    if args.spatial_topk <= 0:
        raise ValueError("--spatial_topk must be positive.")
    if args.classifier_mode not in ("spatial", "mc"):
        raise ValueError("--classifier_mode must be 'spatial' or 'mc'.")
    if args.encoder_hidden_channels <= 0 or args.encoder_out_channels <= 0:
        raise ValueError("Encoder channel counts must be positive.")
    if args.encoder_kernel_size <= 0 or args.encoder_kernel_size % 2 == 0:
        raise ValueError("--encoder_kernel_size must be a positive odd integer.")
    if args.encoder_microbatch_size <= 0 or args.active_map_chunk_size <= 0:
        raise ValueError("Encoder microbatch and active chunk sizes must be positive.")
    if args.mc_channel_epsilon < 0.0:
        raise ValueError("--mc_channel_epsilon must be non-negative.")
    if args.train_feature_source not in ("cache", "online"):
        raise ValueError("--train_feature_source must be 'cache' or 'online'.")
    if args.train_feature_source == "online" and getattr(args, "dataset_tag", None) != "cub200":
        raise ValueError("--train_feature_source online is currently supported only for CUB-200.")
    if args.cub_final_protocol not in ("holdout_best", "full_train_last"):
        raise ValueError("--cub_final_protocol must be 'holdout_best' or 'full_train_last'.")
    if args.cub_final_protocol == "full_train_last" and getattr(args, "dataset_tag", None) != "cub200":
        raise ValueError("--cub_final_protocol full_train_last is supported only for CUB-200.")
    online_aug = bool(args.online_train_augmentation)
    if args.train_feature_source == "online" and not online_aug:
        raise ValueError("--train_feature_source online requires --online_train_augmentation true.")
    if args.train_feature_source == "cache" and online_aug:
        raise ValueError("--online_train_augmentation true requires --train_feature_source online.")
    if args.search_train_per_class < 0 or args.search_val_per_class <= 0 or args.final_val_per_class <= 0:
        raise ValueError("Split sizes must be non-negative for search_train and positive for val splits.")
    if args.overwrite_split and not args.overwrite_subsets:
        print("[Args] --overwrite_split true implies --overwrite_subsets true")
        args.overwrite_subsets = True

    output_dir = Path(args.output_dir)
    if args.feature_dir is None:
        args.feature_dir = str(output_dir / "features")
    if args.selection_json is None:
        args.selection_json = str(output_dir / "selection.json")
    output_dir.mkdir(parents=True, exist_ok=True)
    Path(args.feature_dir).mkdir(parents=True, exist_ok=True)
    return args


def collect_labels(feature_root: Path, split: str) -> np.ndarray:
    stream = FeatureShardStream(feature_root, split)
    labels = np.empty(stream.num_samples, dtype=np.int64)
    seen = np.zeros(stream.num_samples, dtype=np.bool_)
    for shard in stream.shards:
        index_obj = torch_load(stream.split_dir / shard["index_file"], map_location="cpu")
        shard_indices = index_obj["indices"].numpy()
        shard_labels = index_obj["labels"].numpy()
        labels[shard_indices] = shard_labels
        seen[shard_indices] = True
    if not seen.all():
        raise RuntimeError(f"Index files for split {split!r} are incomplete; missing {(~seen).sum()} samples.")
    return labels


def create_or_load_protocol_splits(feature_root: Path, args: argparse.Namespace) -> Dict[str, np.ndarray]:
    dataset_tag = str(getattr(args, "dataset_tag", "dataset"))
    split_path = Path(args.split_file) if args.split_file else Path(args.output_dir) / "splits" / (
        f"{dataset_tag}_protocol_"
        f"seed{args.split_seed}_"
        f"search{args.search_train_per_class}pc_"
        f"searchval{args.search_val_per_class}pc_"
        f"finalval{args.final_val_per_class}pc.npz"
    )
    if split_path.exists() and not args.overwrite_split:
        data = np.load(split_path)
        print(f"[Split] loaded {split_path}")
        return {
            "search_train_indices": data["search_train_indices"],
            "search_val_indices": data["search_val_indices"],
            "final_holdout_val_indices": data["final_holdout_val_indices"],
            "final_train_indices": data["final_train_indices"],
        }

    labels = collect_labels(feature_root, "train")
    num_classes = int(labels.max()) + 1
    rng = np.random.default_rng(args.split_seed)
    search_train_parts: List[np.ndarray] = []
    search_val_parts: List[np.ndarray] = []
    final_holdout_parts: List[np.ndarray] = []

    for class_id in range(num_classes):
        class_indices = np.flatnonzero(labels == class_id).astype(np.int64)
        needed_val = int(args.search_val_per_class) + int(args.final_val_per_class)
        if class_indices.size <= needed_val:
            raise ValueError(
                f"Class {class_id} has only {class_indices.size} samples; need more than "
                f"search_val={args.search_val_per_class} + final_val={args.final_val_per_class}."
            )
        rng.shuffle(class_indices)
        start = 0
        end = start + int(args.search_val_per_class)
        search_val_parts.append(np.sort(class_indices[start:end]))
        start = end
        end = start + int(args.final_val_per_class)
        final_holdout_parts.append(np.sort(class_indices[start:end]))
        remaining = class_indices[end:]
        if int(args.search_train_per_class) > 0:
            if remaining.size < int(args.search_train_per_class):
                raise ValueError(
                    f"Class {class_id} has only {remaining.size} remaining samples for search_train; "
                    f"requested {args.search_train_per_class}."
                )
            search_train_parts.append(np.sort(remaining[: int(args.search_train_per_class)]))
        else:
            search_train_parts.append(np.sort(remaining))

    search_train_indices = np.concatenate(search_train_parts).astype(np.int64)
    search_val_indices = np.concatenate(search_val_parts).astype(np.int64)
    final_holdout_val_indices = np.concatenate(final_holdout_parts).astype(np.int64)
    final_train_mask = np.ones(labels.shape[0], dtype=np.bool_)
    final_train_mask[final_holdout_val_indices] = False
    final_train_indices = np.flatnonzero(final_train_mask).astype(np.int64)

    if search_train_indices.size == 0:
        raise RuntimeError("search_train split is empty.")

    split_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        split_path,
        search_train_indices=search_train_indices,
        search_val_indices=search_val_indices,
        final_holdout_val_indices=final_holdout_val_indices,
        final_train_indices=final_train_indices,
        seed=int(args.split_seed),
        search_train_per_class=int(args.search_train_per_class),
        search_val_per_class=int(args.search_val_per_class),
        final_val_per_class=int(args.final_val_per_class),
        num_classes=int(num_classes),
    )
    print(
        f"[Split] wrote {split_path}: search_train={search_train_indices.size}, "
        f"search_val={search_val_indices.size}, final_holdout_val={final_holdout_val_indices.size}, "
        f"final_train={final_train_indices.size}"
    )
    return {
        "search_train_indices": search_train_indices,
        "search_val_indices": search_val_indices,
        "final_holdout_val_indices": final_holdout_val_indices,
        "final_train_indices": final_train_indices,
    }


def ensure_protocol_subsets(feature_root: Path, args: argparse.Namespace) -> Dict[str, np.ndarray]:
    splits = create_or_load_protocol_splits(feature_root, args)
    verify_split_hashes(
        getattr(args, "expected_split_hashes", None),
        splits,
        "the configured split hashes",
    )
    materialize_train_subset(feature_root, "search_train", splits["search_train_indices"], args)
    materialize_train_subset(feature_root, "search_val", splits["search_val_indices"], args)
    materialize_train_subset(feature_root, "final_holdout_val", splits["final_holdout_val_indices"], args)
    return splits


def run_prepare(args: argparse.Namespace, dataset_factory: DatasetFactory) -> None:
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")
    backbone = create_resnet_backbone(
        args.use_clip,
        args.clip_variant,
        args.clip_pretrained,
        finetuned_model_path=getattr(args, "finetuned_model_path", None),
    ).to(device)
    train_dataset, test_dataset, num_classes, class_to_idx = dataset_factory(args, backbone)
    print(
        f"[Data] {getattr(args, 'dataset_tag', 'dataset')}: "
        f"train={len(train_dataset)}, test={len(test_dataset)}, classes={num_classes}"
    )

    sae = create_image_sae(args.sae_hidden_dim, args.sae_k_sparse).to(device)
    load_image_sae_weights(Path(args.sae_checkpoint), sae, device)
    sae.eval()

    feature_root = Path(args.feature_dir)
    extract_split_to_shards("train", train_dataset, class_to_idx, feature_root, backbone, sae, device, args)
    extract_split_to_shards("test", test_dataset, class_to_idx, feature_root, backbone, sae, device, args)


def run_select(args: argparse.Namespace) -> None:
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    feature_root = Path(args.feature_dir)
    splits = ensure_protocol_subsets(feature_root, args)
    search_train_stream = FeatureShardStream(feature_root, "search_train")
    search_val_stream = FeatureShardStream(feature_root, "search_val")
    args.spatial_dim = int(search_train_stream.spatial_dim)

    lr_grid = parse_float_grid_or_single(args.lr_grid, args.lr)
    weight_decay_grid = parse_float_grid_or_single(args.weight_decay_grid, args.weight_decay)
    batch_size_head_grid = parse_int_grid_or_single(args.batch_size_head_grid, args.batch_size_head)
    sgd_momentum_grid = parse_float_grid_or_single(args.sgd_momentum_grid, args.sgd_momentum)
    sgd_nesterov_grid = parse_bool_grid_or_single(args.sgd_nesterov_grid, args.sgd_nesterov)
    seed_grid = parse_seed_grid_or_single(args.seed_grid, args.seed)
    num_trials = (
        len(lr_grid)
        * len(weight_decay_grid)
        * len(batch_size_head_grid)
        * len(sgd_momentum_grid)
        * len(sgd_nesterov_grid)
        * len(seed_grid)
    )
    num_classes = len(search_train_stream.manifest["classes"])
    trials: List[Dict[str, Any]] = []
    best_trial: Optional[Dict[str, Any]] = None
    trial_id = 0

    for lr in lr_grid:
        for weight_decay in weight_decay_grid:
            for batch_size_head in batch_size_head_grid:
                for sgd_momentum in sgd_momentum_grid:
                    for sgd_nesterov in sgd_nesterov_grid:
                        for trial_seed in seed_grid:
                            set_seed(int(trial_seed))
                            trial_args = argparse.Namespace(**vars(args))
                            trial_args.lr = float(lr)
                            trial_args.weight_decay = float(weight_decay)
                            trial_args.batch_size_head = int(batch_size_head)
                            trial_args.sgd_momentum = float(sgd_momentum)
                            trial_args.sgd_nesterov = bool(sgd_nesterov)
                            trial_args.seed = int(trial_seed)
                            classifier = build_classifier(trial_args, search_train_stream.feature_dim, num_classes, device)
                            checkpoint_path = None
                            if args.save_select_checkpoints:
                                checkpoint_path = Path(args.output_dir) / "checkpoints" / f"select_trial_{trial_id:03d}.pth"
                            print(
                                f"[Select] trial {trial_id + 1}/{num_trials}: "
                                f"lr={lr:.3g}, weight_decay={weight_decay:.3g}, "
                                f"batch_size_head={batch_size_head}, "
                                f"sgd_momentum={sgd_momentum:.3g}, "
                                f"sgd_nesterov={bool(sgd_nesterov)}, "
                                f"seed={trial_seed}, "
                                f"encoder={args.encoder_hidden_channels}->{args.encoder_out_channels}, "
                                f"kernel={args.encoder_kernel_size}",
                                flush=True,
                            )
                            best_acc, best_epoch, train_losses, val_accs = train_with_validation(
                                classifier=classifier,
                                train_stream=search_train_stream,
                                val_stream=search_val_stream,
                                train_mask=None,
                                val_mask=None,
                                args=trial_args,
                                device=device,
                                checkpoint_path=checkpoint_path,
                                epochs=args.search_epochs,
                                desc_prefix=f"Select {trial_id + 1}/{num_trials}",
                            )
                            trial = {
                                "trial_id": int(trial_id),
                                "hparams": {
                                    "optimizer": args.optimizer,
                                    "lr": float(lr),
                                    "weight_decay": float(weight_decay),
                                    "sgd_momentum": float(sgd_momentum),
                                    "sgd_nesterov": bool(sgd_nesterov),
                                    "classifier_mode": str(args.classifier_mode),
                                    "encoder_hidden_channels": int(args.encoder_hidden_channels),
                                    "encoder_out_channels": int(args.encoder_out_channels),
                                    "encoder_kernel_size": int(args.encoder_kernel_size),
                                    "encoder_microbatch_size": int(args.encoder_microbatch_size),
                                    "active_map_chunk_size": int(args.active_map_chunk_size),
                                    "spatial_topk": int(args.spatial_topk),
                                    "mc_channel_epsilon": float(args.mc_channel_epsilon),
                                    "mc_sqrt_count": bool(args.mc_sqrt_count),
                                    "mc_blocknorm": str(args.mc_blocknorm),
                                    "batch_size_head": int(batch_size_head),
                                    "seed": int(trial_seed),
                                },
                                "best_search_val_acc": float(best_acc),
                                "best_search_epoch": int(best_epoch),
                                "search_epochs": int(args.search_epochs),
                                "train_losses": train_losses,
                                "search_val_accs": val_accs,
                                "checkpoint": str(checkpoint_path) if checkpoint_path is not None else None,
                            }
                            trials.append(trial)
                            if best_trial is None or trial["best_search_val_acc"] > best_trial["best_search_val_acc"]:
                                best_trial = trial

                            write_json(
                                Path(args.output_dir) / "selection_progress.json",
                                {
                                    "completed_trials": len(trials),
                                    "num_trials": int(num_trials),
                                    "grid": {
                                        "lr": [float(v) for v in lr_grid],
                                        "weight_decay": [float(v) for v in weight_decay_grid],
                                        "batch_size_head": [int(v) for v in batch_size_head_grid],
                                        "sgd_momentum": [float(v) for v in sgd_momentum_grid],
                                        "sgd_nesterov": [bool(v) for v in sgd_nesterov_grid],
                                        "seed": [int(v) for v in seed_grid],
                                    },
                                    "best_trial": best_trial,
                                    "trials": trials,
                                },
                            )
                            trial_id += 1

    if best_trial is None:
        raise RuntimeError("No selection trials were run.")

    selection = {
        "protocol": "small_dataset_subset_hparam_search",
        "dataset": getattr(args, "dataset_tag", "dataset"),
        "best_hparams": best_trial["hparams"],
        "best_search_val_acc": best_trial["best_search_val_acc"],
        "best_search_epoch": best_trial["best_search_epoch"],
        "search_epochs": int(args.search_epochs),
        "search_train_size": int(splits["search_train_indices"].size),
        "search_val_size": int(splits["search_val_indices"].size),
        "final_holdout_val_size": int(splits["final_holdout_val_indices"].size),
        "final_train_size": int(splits["final_train_indices"].size),
        "grid": {
            "lr": [float(v) for v in lr_grid],
            "weight_decay": [float(v) for v in weight_decay_grid],
            "batch_size_head": [int(v) for v in batch_size_head_grid],
            "sgd_momentum": [float(v) for v in sgd_momentum_grid],
            "sgd_nesterov": [bool(v) for v in sgd_nesterov_grid],
            "seed": [int(v) for v in seed_grid],
        },
        "split_indices_sha256": {
            "search_train": indices_sha256(splits["search_train_indices"]),
            "search_val": indices_sha256(splits["search_val_indices"]),
            "final_holdout_val": indices_sha256(splits["final_holdout_val_indices"]),
            "final_train": indices_sha256(splits["final_train_indices"]),
        },
        "trials": trials,
        "args": vars(args),
    }
    selection_path = Path(args.selection_json)
    write_json(selection_path, selection)
    print(f"[Select] wrote {selection_path}")


def apply_best_hparams(args: argparse.Namespace, selection: Dict[str, Any]) -> None:
    hparams = selection.get("best_hparams", {})
    if not hparams:
        raise ValueError("selection.json does not contain best_hparams. Re-run --stage select.")
    for key in [
        "optimizer",
        "lr",
        "weight_decay",
        "sgd_momentum",
        "sgd_nesterov",
        "classifier_mode",
        "encoder_hidden_channels",
        "encoder_out_channels",
        "encoder_kernel_size",
        "encoder_microbatch_size",
        "active_map_chunk_size",
        "spatial_topk",
        "mc_channel_epsilon",
        "mc_sqrt_count",
        "mc_blocknorm",
        "batch_size_head",
        "seed",
    ]:
        if key in hparams:
            setattr(args, key, hparams[key])


def train_without_validation(
    classifier: torch.nn.Module,
    train_stream: Any,
    train_mask: Optional[np.ndarray],
    args: argparse.Namespace,
    device: torch.device,
    checkpoint_path: Path,
    epochs: int,
    desc_prefix: str,
    scheduler_t_max: Optional[int] = None,
) -> Tuple[List[float], List[float]]:
    fit_normalizer = getattr(classifier, "fit_normalizer", None)
    if callable(fit_normalizer):
        fit_normalizer(
            train_stream,
            train_mask,
            batch_size=args.batch_size_head,
            device=device,
        )

    optimizer = build_optimizer(args, classifier.parameters())
    criterion = torch.nn.CrossEntropyLoss()
    effective_t_max = int(scheduler_t_max) if scheduler_t_max is not None else int(epochs)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(effective_t_max, 1))
    train_losses: List[float] = []
    train_accs: List[float] = []

    for epoch in range(int(epochs)):
        classifier.train()
        loss_sum = 0.0
        correct = 0
        total = 0
        batch_count = 0

        batches = train_stream.iter_batches(
            batch_size=args.batch_size_head,
            allowed_mask=train_mask,
            shuffle=True,
            seed=args.seed + epoch,
        )
        for concept_indices, concept_values, labels in batches:
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            batch_n = int(labels.numel())
            micro = max(int(getattr(classifier, "train_microbatch_size", args.batch_size_head)), 1)
            batch_loss_sum = 0.0
            batch_correct = 0
            for start in range(0, batch_n, micro):
                end = min(start + micro, batch_n)
                idx_mb = concept_indices[start:end].to(device, non_blocking=True)
                val_mb = concept_values[start:end].to(device, non_blocking=True)
                labels_mb = labels[start:end]
                logits = classifier(idx_mb, val_mb)
                loss = criterion(logits, labels_mb)
                scaled_loss = loss * (float(labels_mb.numel()) / float(max(batch_n, 1)))
                scaled_loss.backward()
                batch_loss_sum += float(loss.item()) * int(labels_mb.numel())
                pred = logits.argmax(dim=1)
                batch_correct += int(pred.eq(labels_mb).sum().item())
            optimizer.step()

            loss_sum += batch_loss_sum / float(max(batch_n, 1))
            batch_count += 1
            total += int(labels.numel())
            correct += batch_correct

        train_loss = loss_sum / max(batch_count, 1)
        train_acc = 100.0 * correct / max(total, 1)
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        scheduler.step()
        lr = scheduler.get_last_lr()[0]
        print(
            f"[{desc_prefix} epoch {epoch + 1}/{epochs}] train_loss={train_loss:.4f} "
            f"train_acc={train_acc:.2f}% lr={lr:.2e}",
            flush=True,
        )

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": int(epochs) - 1,
            "model_state_dict": classifier.state_dict(),
            "train_loss": train_losses[-1] if train_losses else None,
            "train_acc": train_accs[-1] if train_accs else None,
            "hparams": {
                "optimizer": args.optimizer,
                "lr": float(args.lr),
                "weight_decay": float(args.weight_decay),
                "sgd_momentum": float(args.sgd_momentum),
                "sgd_nesterov": bool(args.sgd_nesterov),
                "classifier_mode": str(getattr(args, "classifier_mode", "spatial")),
                "encoder_hidden_channels": int(args.encoder_hidden_channels),
                "encoder_out_channels": int(args.encoder_out_channels),
                "encoder_kernel_size": int(args.encoder_kernel_size),
                "encoder_microbatch_size": int(args.encoder_microbatch_size),
                "active_map_chunk_size": int(args.active_map_chunk_size),
                "spatial_topk": int(args.spatial_topk),
                "mc_channel_epsilon": float(getattr(args, "mc_channel_epsilon", 1e-4)),
                "mc_sqrt_count": bool(getattr(args, "mc_sqrt_count", True)),
                "mc_blocknorm": str(getattr(args, "mc_blocknorm", "stat")),
                "batch_size_head": int(args.batch_size_head),
                "seed": int(args.seed),
            },
            "args": vars(args),
        },
        checkpoint_path,
    )
    print(f"[{desc_prefix}] wrote last checkpoint: {checkpoint_path}", flush=True)
    return train_losses, train_accs


def run_cub_full_train_last(
    args: argparse.Namespace,
    dataset_factory: Optional[DatasetFactory],
    selection: Dict[str, Any],
    feature_root: Path,
    selection_path: Path,
    device: torch.device,
) -> None:
    if dataset_factory is None:
        raise RuntimeError("CUB full_train_last final requires a dataset_factory.")
    if str(getattr(args, "dataset_tag", "")) != "cub200":
        raise ValueError("full_train_last is supported only for CUB-200.")
    if str(getattr(args, "train_feature_source", "cache")) != "online":
        raise ValueError("full_train_last requires --train_feature_source online.")
    if not bool(getattr(args, "online_train_augmentation", False)):
        raise ValueError("full_train_last requires --online_train_augmentation true.")

    train_cache_stream = FeatureShardStream(feature_root, "train")
    test_stream = FeatureShardStream(feature_root, "test")
    args.spatial_dim = int(train_cache_stream.spatial_dim)
    full_train_indices = np.arange(train_cache_stream.num_samples, dtype=np.int64)
    train_classes = list(train_cache_stream.manifest["classes"])

    print(
        "[Final] cub_final_protocol=full_train_last; "
        "checkpoint_selection=last_epoch_no_validation",
        flush=True,
    )
    backbone = create_resnet_backbone(
        args.use_clip,
        args.clip_variant,
        args.clip_pretrained,
        finetuned_model_path=getattr(args, "finetuned_model_path", None),
    ).to(device)
    sae = create_image_sae(args.sae_hidden_dim, args.sae_k_sparse).to(device)
    load_image_sae_weights(Path(args.sae_checkpoint), sae, device)
    sae.eval()

    dataset_args = argparse.Namespace(**vars(args))
    dataset_args._cub_apply_train_augmentation = True
    train_dataset, _test_dataset, online_num_classes, class_to_idx = dataset_factory(dataset_args, backbone)
    expected_class_to_idx = {str(cls): idx for idx, cls in enumerate(train_classes)}
    if {str(k): int(v) for k, v in class_to_idx.items()} != expected_class_to_idx:
        raise ValueError("Online CUB dataset class_to_idx does not match cached train manifest.")
    if int(online_num_classes) != len(train_classes):
        raise ValueError(
            f"Online dataset class count {online_num_classes} does not match cached train classes {len(train_classes)}."
        )
    if len(train_dataset) != int(train_cache_stream.num_samples):
        raise ValueError(
            f"Online train dataset size {len(train_dataset)} does not match cached train size "
            f"{train_cache_stream.num_samples}."
        )

    train_stream = OnlineFeatureStream(
        dataset=train_dataset,
        sample_indices=full_train_indices,
        classes=train_classes,
        backbone=backbone,
        sae=sae,
        device=device,
        args=args,
        split="full_train_online",
    )
    num_classes = len(train_stream.manifest["classes"])
    classifier = build_classifier(args, train_stream.feature_dim, num_classes, device)
    final_ckpt = Path(args.output_dir) / "checkpoints" / "final_last.pth"
    print(
        f"[Final] epochs={args.final_epochs}, scheduler_t_max={args.final_scheduler_t_max}, "
        f"train={train_stream.num_samples} (full_train_online), holdout_val=0",
        flush=True,
    )
    train_losses, train_accs = train_without_validation(
        classifier=classifier,
        train_stream=train_stream,
        train_mask=None,
        args=args,
        device=device,
        checkpoint_path=final_ckpt,
        epochs=args.final_epochs,
        desc_prefix="FinalFullTrain",
        scheduler_t_max=int(args.final_scheduler_t_max),
    )

    final_payload = torch.load(final_ckpt, map_location=device, weights_only=False)
    classifier.load_state_dict(final_payload["model_state_dict"])
    test_acc = evaluate_stream(
        classifier,
        test_stream,
        allowed_mask=None,
        batch_size=args.batch_size_head,
        device=device,
    )
    print(f"[Final] test_acc={test_acc:.2f}%")

    result = {
        "dataset": getattr(args, "dataset_tag", "dataset"),
        "protocol": "cub_full_train_last_checkpoint_test_once",
        "checkpoint_selection": "last_epoch_no_validation",
        "best_final_holdout_acc": None,
        "best_final_epoch": None,
        "final_epoch": int(args.final_epochs) - 1,
        "final_last_train_loss": train_losses[-1] if train_losses else None,
        "final_last_train_acc": train_accs[-1] if train_accs else None,
        "test_acc": float(test_acc),
        "final_epochs": int(args.final_epochs),
        "final_scheduler_t_max": int(args.final_scheduler_t_max),
        "selection_json": str(selection_path),
        "final_checkpoint": str(final_ckpt),
        "train_losses": train_losses,
        "train_accs": train_accs,
        "best_hparams": selection["best_hparams"],
        "search_summary": {
            "best_search_val_acc": selection.get("best_search_val_acc"),
            "best_search_epoch": selection.get("best_search_epoch"),
            "search_train_size": selection.get("search_train_size"),
            "search_val_size": selection.get("search_val_size"),
        },
        "split_sizes": {
            "final_train": int(train_stream.num_samples),
            "final_holdout_val": 0,
            "test": int(test_stream.num_samples),
        },
        "feature_dir": str(feature_root),
        "train_feature_source": "online",
        "online_train_augmentation": True,
        "cub_final_protocol": "full_train_last",
        "args": vars(args),
    }
    result_path = Path(args.output_dir) / f"{getattr(args, 'dataset_tag', 'dataset')}_spatial_encoder_results.json"
    write_json(result_path, result)
    print(f"[Final] wrote {result_path}")


def run_final(args: argparse.Namespace, dataset_factory: Optional[DatasetFactory] = None) -> None:
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    feature_root = Path(args.feature_dir)
    selection_path = Path(args.selection_json)
    if not selection_path.exists():
        raise FileNotFoundError(f"Missing selection json for final stage: {selection_path}")
    selection = read_json(selection_path)
    splits = ensure_protocol_subsets(feature_root, args)
    verify_selection_splits(selection, splits)
    apply_best_hparams(args, selection)
    set_seed(args.seed)

    if str(getattr(args, "cub_final_protocol", "holdout_best")) == "full_train_last":
        run_cub_full_train_last(
            args=args,
            dataset_factory=dataset_factory,
            selection=selection,
            feature_root=feature_root,
            selection_path=selection_path,
            device=device,
        )
        return

    final_holdout_val_stream = FeatureShardStream(feature_root, "final_holdout_val")
    test_stream = FeatureShardStream(feature_root, "test")
    args.spatial_dim = int(final_holdout_val_stream.spatial_dim)
    train_source = str(getattr(args, "train_feature_source", "cache"))

    if train_source == "cache":
        materialize_train_subset(feature_root, "final_train", splits["final_train_indices"], args)
        train_stream = FeatureShardStream(feature_root, "final_train")
        train_desc = "final_train"
    elif train_source == "online":
        if dataset_factory is None:
            raise RuntimeError("Online final training requires a dataset_factory.")
        print(
            "[Final] train_feature_source=online; "
            f"online_train_augmentation={bool(args.online_train_augmentation)}",
            flush=True,
        )
        backbone = create_resnet_backbone(
            args.use_clip,
            args.clip_variant,
            args.clip_pretrained,
            finetuned_model_path=getattr(args, "finetuned_model_path", None),
        ).to(device)
        sae = create_image_sae(args.sae_hidden_dim, args.sae_k_sparse).to(device)
        load_image_sae_weights(Path(args.sae_checkpoint), sae, device)
        sae.eval()

        dataset_args = argparse.Namespace(**vars(args))
        dataset_args._cub_apply_train_augmentation = True
        train_dataset, _test_dataset, online_num_classes, _class_to_idx = dataset_factory(dataset_args, backbone)
        holdout_classes = list(final_holdout_val_stream.manifest["classes"])
        if int(online_num_classes) != len(holdout_classes):
            raise ValueError(
                f"Online dataset class count {online_num_classes} does not match "
                f"cached holdout classes {len(holdout_classes)}."
            )
        train_stream = OnlineFeatureStream(
            dataset=train_dataset,
            sample_indices=splits["final_train_indices"],
            classes=holdout_classes,
            backbone=backbone,
            sae=sae,
            device=device,
            args=args,
            split="final_train_online",
        )
        train_desc = "final_train_online"
    else:
        raise ValueError(f"Unsupported train_feature_source: {train_source!r}")

    num_classes = len(train_stream.manifest["classes"])
    classifier = build_classifier(args, train_stream.feature_dim, num_classes, device)
    final_ckpt = Path(args.output_dir) / "checkpoints" / "final_best.pth"
    print(
        f"[Final] epochs={args.final_epochs}, scheduler_t_max={args.final_scheduler_t_max}, "
        f"train={train_stream.num_samples} ({train_desc}), "
        f"holdout_val={splits['final_holdout_val_indices'].size}",
        flush=True,
    )
    best_holdout_acc, best_epoch, train_losses, holdout_val_accs = train_with_validation(
        classifier=classifier,
        train_stream=train_stream,
        val_stream=final_holdout_val_stream,
        train_mask=None,
        val_mask=None,
        args=args,
        device=device,
        checkpoint_path=final_ckpt,
        epochs=args.final_epochs,
        desc_prefix="Final",
        scheduler_t_max=int(args.final_scheduler_t_max),
    )

    best_payload = torch.load(final_ckpt, map_location=device, weights_only=False)
    classifier.load_state_dict(best_payload["model_state_dict"])
    test_acc = evaluate_stream(
        classifier,
        test_stream,
        allowed_mask=None,
        batch_size=args.batch_size_head,
        device=device,
    )
    print(f"[Final] test_acc={test_acc:.2f}%")

    result = {
        "dataset": getattr(args, "dataset_tag", "dataset"),
        "protocol": "train_internal_holdout_select_test_once",
        "best_final_holdout_acc": float(best_holdout_acc),
        "best_final_epoch": int(best_epoch),
        "test_acc": float(test_acc),
        "final_epochs": int(args.final_epochs),
        "final_scheduler_t_max": int(args.final_scheduler_t_max),
        "selection_json": str(selection_path),
        "final_checkpoint": str(final_ckpt),
        "train_losses": train_losses,
        "final_holdout_val_accs": holdout_val_accs,
        "best_hparams": selection["best_hparams"],
        "search_summary": {
            "best_search_val_acc": selection.get("best_search_val_acc"),
            "best_search_epoch": selection.get("best_search_epoch"),
            "search_train_size": selection.get("search_train_size"),
            "search_val_size": selection.get("search_val_size"),
        },
        "split_sizes": {
            "final_train": int(splits["final_train_indices"].size),
            "final_holdout_val": int(splits["final_holdout_val_indices"].size),
            "test": int(test_stream.num_samples),
        },
        "feature_dir": str(feature_root),
        "train_feature_source": train_source,
        "online_train_augmentation": bool(getattr(args, "online_train_augmentation", False)),
        "args": vars(args),
    }
    result_path = Path(args.output_dir) / f"{getattr(args, 'dataset_tag', 'dataset')}_spatial_encoder_results.json"
    write_json(result_path, result)
    print(f"[Final] wrote {result_path}")


def run_stage(args: argparse.Namespace, dataset_factory: DatasetFactory) -> None:
    set_seed(args.seed)
    print({k: v for k, v in vars(args).items() if k != "dataset_factory"}, flush=True)
    if args.stage in ("prepare", "all"):
        run_prepare(args, dataset_factory)
    if args.stage in ("select", "all"):
        run_select(args)
    if args.stage in ("final", "all"):
        run_final(args, dataset_factory)
