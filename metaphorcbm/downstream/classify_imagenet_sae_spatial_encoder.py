#!/usr/bin/env python3
"""ImageNet-1K classification with frozen CLIP-RN50 and ImageSAE features.

The pipeline caches sparse 7x7 ImageSAE activation maps and trains a shared
2D encoder:

  f_theta: R^{1 x 7 x 7} -> R^K

which is applied independently to each concept. The resulting features feed
a linear classification head.

Features are cached as shards and read as streams during training and
evaluation.

Protocol:
  - prepare: extract full official train and official val sparse spatial shards
  - select: materialize small disjoint train subsets for hparam search
  - final: train on official train minus a held-out split, select the best
    checkpoint on that held-out split, then evaluate official val once as test
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from PIL import ImageFile
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
from tqdm import tqdm

from metaphorcbm.config import parse_args_with_config

ImageFile.LOAD_TRUNCATED_IMAGES = True

BOTTLENECK_NAME = "spatial_topk_concept_maps"
MODEL_NAME = "shared_small_2d_spatial_encoder"
FEATURE_LAYOUT = "sparse_patch_topk"
MIXED_FINAL_TRAIN_SPLIT = "final_train_mixed"
FINAL_TRAIN_MIX_BUFFER_SHARDS = 32


def progress_disabled() -> bool:
    flag = os.environ.get("CONCEPT_SAE_PROGRESS", "").strip().lower()
    if flag in {"0", "false", "no", "off", "disable", "disabled"}:
        return True
    if flag in {"1", "true", "yes", "on", "enable", "enabled"}:
        return False
    return not sys.stderr.isatty()


def progress_bar(iterable: Any, **kwargs: Any) -> Any:
    kwargs.setdefault("dynamic_ncols", True)
    kwargs.setdefault("disable", progress_disabled())
    return tqdm(iterable, **kwargs)


from metaphorcbm.models import ImageSAE as FrameworkImageSAE
from metaphorcbm.models import ResNetWithHooks, create_clip_resnet_backbone


def str2bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    if v is None:
        return False
    s = str(v).strip().lower()
    if s in ("1", "true", "t", "yes", "y", "on"):
        return True
    if s in ("0", "false", "f", "no", "n", "off"):
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {v}")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    tmp.replace(path)


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def torch_load(path: Path, map_location: str | torch.device = "cpu") -> Any:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def create_resnet_backbone(
    use_clip: bool = True,
    clip_variant: str = "RN50",
    clip_pretrained: str = "openai",
    finetuned_model_path: Optional[str] = None,
) -> nn.Module:
    finetuned_model_path = finetuned_model_path.strip() if finetuned_model_path else None
    if use_clip:
        print(f"[Backbone] CLIP-{clip_variant} ({clip_pretrained})")
        backbone = create_clip_resnet_backbone(
            {
                "clip_variant": clip_variant,
                "pretrained": clip_pretrained,
                "finetuned_model_path": finetuned_model_path,
                "hook_layer": "layer4",
                "freeze_backbone": True,
                "return_features": False,
            }
        )
    else:
        print("[Backbone] torchvision ResNet-50 IMAGENET1K_V2")
        backbone = ResNetWithHooks(
            weights="IMAGENET1K_V2",
            hook_layer="layer4",
            freeze_backbone=True,
            return_features=False,
        )
    backbone.eval()
    return backbone


def create_image_sae(hidden_dim: int = 8192, k_sparse: int = 64) -> nn.Module:
    sae = FrameworkImageSAE(
        input_dim=2048,
        hidden_dim=hidden_dim,
        k_sparse=k_sparse,
        use_bias=False,
        activation="relu",
        initialization="fan_in",
    )
    print(f"[SAE] hidden_dim={hidden_dim}, k_sparse={k_sparse}")
    return sae


def _strip_prefix_state_dict(state: Dict[str, torch.Tensor], prefix: str) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    for key, value in state.items():
        if key.startswith(prefix):
            out[key[len(prefix) :]] = value
    return out


def load_image_sae_weights(checkpoint_path: Path, image_sae: nn.Module, device: torch.device) -> None:
    print(f"[SAE] Loading checkpoint: {checkpoint_path}")
    checkpoint = torch_load(checkpoint_path, map_location=device)

    candidates: List[Dict[str, torch.Tensor]] = []
    if isinstance(checkpoint, dict):
        model_state = checkpoint.get("model_state_dict")
        if isinstance(model_state, dict):
            if isinstance(model_state.get("image_sae"), dict):
                candidates.append(model_state["image_sae"])
            stripped = _strip_prefix_state_dict(model_state, "image_sae.")
            if stripped:
                candidates.append(stripped)
            candidates.append(model_state)
        if isinstance(checkpoint.get("image_sae_state_dict"), dict):
            candidates.append(checkpoint["image_sae_state_dict"])
        candidates.append(checkpoint)

    last_error: Optional[Exception] = None
    target_keys = set(image_sae.state_dict().keys())
    for state in candidates:
        if not (target_keys & set(state.keys())):
            continue
        try:
            image_sae.load_state_dict(state, strict=True)
            encoder_weight = (
                image_sae.encoder[-1].weight
                if hasattr(image_sae.encoder, "__getitem__")
                else image_sae.encoder.weight
            )
            print(f"[SAE] loaded. encoder weight shape={tuple(encoder_weight.shape)}")
            return
        except Exception as exc:
            last_error = exc

    raise RuntimeError(f"Could not load ImageSAE weights from {checkpoint_path}: {last_error}")


class SpatialConceptClassifier(nn.Module):
    """Concept-preserving learned spatial aggregation plus linear class head."""

    def __init__(
        self,
        hidden_dim: int,
        spatial_dim: int,
        encoder_hidden_channels: int,
        encoder_out_channels: int,
        encoder_kernel_size: int,
        num_classes: int,
        active_chunk_size: int,
    ):
        super().__init__()
        side = int(round(float(spatial_dim) ** 0.5))
        if side * side != int(spatial_dim):
            raise ValueError(f"Expected a square spatial grid, got spatial_dim={spatial_dim}.")
        self.hidden_dim = int(hidden_dim)
        self.spatial_dim = int(spatial_dim)
        self.spatial_side = int(side)
        self.encoder_hidden_channels = int(encoder_hidden_channels)
        self.encoder_out_channels = int(encoder_out_channels)
        self.encoder_kernel_size = int(encoder_kernel_size)
        if self.encoder_kernel_size <= 0 or self.encoder_kernel_size % 2 == 0:
            raise ValueError(f"encoder_kernel_size must be a positive odd integer, got {encoder_kernel_size}.")
        self.active_chunk_size = int(active_chunk_size)

        self.encoder_conv = nn.Conv2d(
            1,
            self.encoder_hidden_channels,
            kernel_size=self.encoder_kernel_size,
            padding=self.encoder_kernel_size // 2,
            bias=False,
        )
        self.encoder_activation = nn.GELU()
        self.encoder_proj = nn.Linear(self.encoder_hidden_channels, self.encoder_out_channels, bias=False)
        self.classifier = nn.Linear(self.hidden_dim * self.encoder_out_channels, num_classes)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_normal_(self.encoder_conv.weight, nonlinearity="linear")
        nn.init.xavier_uniform_(self.encoder_proj.weight)
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def _active_maps(
        self,
        concept_indices: torch.Tensor,
        concept_values: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bsz, spatial_dim, topk = concept_indices.shape
        if int(spatial_dim) != self.spatial_dim:
            raise ValueError(f"Expected spatial_dim={self.spatial_dim}, got {spatial_dim}.")
        device = concept_values.device
        sample_ids = torch.arange(bsz, device=device, dtype=torch.long).view(bsz, 1, 1).expand(-1, spatial_dim, topk)
        patch_ids = torch.arange(spatial_dim, device=device, dtype=torch.long).view(1, spatial_dim, 1).expand(bsz, -1, topk)
        concept_ids = concept_indices.to(device=device, dtype=torch.long)
        values = concept_values.to(device=device, dtype=torch.float32)

        flat_values = values.reshape(-1)
        keep = flat_values > 0
        if not bool(keep.any()):
            empty_maps = values.new_zeros((0, self.spatial_dim))
            empty_ids = concept_ids.new_zeros((0,))
            return empty_maps, empty_ids, empty_ids

        flat_samples = sample_ids.reshape(-1)[keep]
        flat_patches = patch_ids.reshape(-1)[keep]
        flat_concepts = concept_ids.reshape(-1)[keep]
        flat_values = flat_values[keep]

        keys = flat_samples * self.hidden_dim + flat_concepts
        unique_keys, inverse = torch.unique(keys, sorted=False, return_inverse=True)
        active_samples = torch.div(unique_keys, self.hidden_dim, rounding_mode="floor")
        active_concepts = unique_keys.remainder(self.hidden_dim)
        maps = flat_values.new_zeros((int(unique_keys.numel()), self.spatial_dim))
        maps.index_put_((inverse, flat_patches), flat_values, accumulate=True)
        return maps, active_samples, active_concepts

    def _encode_maps(self, maps: torch.Tensor) -> torch.Tensor:
        if maps.numel() == 0:
            return maps.new_zeros((0, self.encoder_out_channels))
        outputs: List[torch.Tensor] = []
        chunk_size = max(int(self.active_chunk_size), 1)
        for start in range(0, int(maps.shape[0]), chunk_size):
            chunk = maps[start : start + chunk_size].view(-1, 1, self.spatial_side, self.spatial_side)
            hidden = self.encoder_activation(self.encoder_conv(chunk))
            pooled = hidden.mean(dim=(2, 3))
            outputs.append(self.encoder_proj(pooled))
        return torch.cat(outputs, dim=0)

    def forward(self, concept_indices: torch.Tensor, concept_values: torch.Tensor) -> torch.Tensor:
        bsz = int(concept_indices.shape[0])
        maps, active_samples, active_concepts = self._active_maps(concept_indices, concept_values)
        active_delta = self._encode_maps(maps)

        weight = self.classifier.weight.view(
            self.classifier.out_features,
            self.hidden_dim,
            self.encoder_out_channels,
        )
        base_logits = self.classifier.bias
        logits = base_logits.view(1, -1).expand(bsz, -1).clone()

        if active_delta.numel() == 0:
            return logits
        chunk_size = max(int(self.active_chunk_size), 1)
        for start in range(0, int(active_delta.shape[0]), chunk_size):
            end = start + chunk_size
            delta = active_delta[start:end]
            concept_chunk = active_concepts[start:end]
            weight_chunk = weight[:, concept_chunk, :]
            contrib = torch.einsum("mk,ymk->my", delta, weight_chunk)
            logits.index_add_(0, active_samples[start:end], contrib)
        return logits


class MCStatsClassifier(nn.Module):
    """Linear classifier on per-concept maximum activations and activation counts."""

    def __init__(
        self,
        hidden_dim: int,
        num_classes: int,
        epsilon: float,
        sqrt_count: bool,
        blocknorm: str,
    ):
        super().__init__()
        if blocknorm not in ("none", "stat"):
            raise ValueError(f"Unsupported mc blocknorm: {blocknorm!r}")
        self.hidden_dim = int(hidden_dim)
        self.epsilon = float(epsilon)
        self.sqrt_count = bool(sqrt_count)
        self.blocknorm = str(blocknorm)
        self.fc = nn.Linear(self.hidden_dim * 2, num_classes)
        self.register_buffer("norm_mu", torch.zeros(2, dtype=torch.float32))
        self.register_buffer("norm_sigma", torch.ones(2, dtype=torch.float32))
        self.register_buffer("norm_fitted", torch.tensor(False, dtype=torch.bool))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def _stats_tensor(
        self,
        concept_indices: torch.Tensor,
        concept_values: torch.Tensor,
        *,
        normalize: bool,
    ) -> torch.Tensor:
        bsz, _spatial_dim, _topk = concept_indices.shape
        device = concept_values.device
        concept_ids = concept_indices.to(device=device, dtype=torch.long)
        values = concept_values.to(device=device, dtype=torch.float32)

        flat_values = values.reshape(-1)
        keep = flat_values > self.epsilon

        max_flat = values.new_zeros((bsz * self.hidden_dim,))
        count_flat = values.new_zeros((bsz * self.hidden_dim,))
        if bool(keep.any()):
            sample_ids = (
                torch.arange(bsz, device=device, dtype=torch.long)
                .view(bsz, 1, 1)
                .expand_as(concept_ids)
                .reshape(-1)[keep]
            )
            flat_concepts = concept_ids.reshape(-1)[keep]
            offsets = sample_ids * self.hidden_dim + flat_concepts
            kept_values = flat_values[keep]
            max_flat.scatter_reduce_(0, offsets, kept_values, reduce="amax", include_self=True)
            count_flat.index_add_(0, offsets, torch.ones_like(kept_values))

        max_vals = max_flat.view(bsz, self.hidden_dim)
        counts = count_flat.view(bsz, self.hidden_dim)
        if self.sqrt_count:
            counts = torch.sqrt(torch.clamp(counts, min=0.0))
        stats = torch.stack([max_vals, counts], dim=2)

        if normalize and self.blocknorm == "stat":
            if not bool(self.norm_fitted.item()):
                raise RuntimeError("MCStatsClassifier blocknorm is enabled but has not been fitted.")
            stats = (stats - self.norm_mu.view(1, 1, 2)) / self.norm_sigma.view(1, 1, 2)
        return stats

    @torch.no_grad()
    def fit_normalizer(
        self,
        stream: "FeatureShardStream",
        allowed_mask: Optional[np.ndarray],
        batch_size: int,
        device: torch.device,
    ) -> None:
        if self.blocknorm == "none":
            self.norm_mu.zero_()
            self.norm_sigma.fill_(1.0)
            self.norm_fitted.fill_(True)
            return

        sum_b = torch.zeros(2, dtype=torch.float64, device=device)
        sqsum_b = torch.zeros(2, dtype=torch.float64, device=device)
        total = 0
        for concept_indices, concept_values, _labels in stream.iter_batches(
            batch_size=batch_size,
            allowed_mask=allowed_mask,
            shuffle=False,
        ):
            stats = self._stats_tensor(
                concept_indices.to(device, non_blocking=True),
                concept_values.to(device, non_blocking=True),
                normalize=False,
            )
            stats64 = stats.to(torch.float64)
            sum_b += stats64.sum(dim=(0, 1))
            sqsum_b += (stats64 * stats64).sum(dim=(0, 1))
            total += int(stats.shape[0]) * int(stats.shape[1])

        denom = max(total, 1)
        mu = sum_b / float(denom)
        var = sqsum_b / float(denom) - mu * mu
        sigma = torch.sqrt(torch.clamp(var, min=1e-12))
        self.norm_mu.copy_(mu.to(torch.float32))
        self.norm_sigma.copy_(torch.clamp(sigma.to(torch.float32), min=1e-8))
        self.norm_fitted.fill_(True)
        print(
            f"[MCStats] fitted blocknorm=stat mu={self.norm_mu.tolist()} "
            f"sigma={self.norm_sigma.tolist()}",
            flush=True,
        )

    def forward(self, concept_indices: torch.Tensor, concept_values: torch.Tensor) -> torch.Tensor:
        stats = self._stats_tensor(concept_indices, concept_values, normalize=True)
        return self.fc(stats.reshape(int(stats.shape[0]), self.hidden_dim * 2))


class IndexedDataset(Dataset):
    def __init__(self, dataset: Dataset, start_index: int = 0):
        self.dataset = dataset
        self.start_index = int(start_index)

    def __len__(self) -> int:
        return len(self.dataset) - self.start_index

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int, int]:
        original_index = self.start_index + int(index)
        image, label = self.dataset[original_index]
        return image, int(label), original_index


def build_imagenet_transform(backbone: nn.Module, use_clip: bool) -> transforms.Compose:
    if use_clip and hasattr(backbone, "preprocess") and backbone.preprocess is not None:
        return backbone.preprocess
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def load_imagenet_folders(data_root: Path, transform: Any) -> Tuple[ImageFolder, ImageFolder]:
    train_root = data_root / "train"
    val_root = data_root / "val"
    if not train_root.is_dir() or not val_root.is_dir():
        raise FileNotFoundError(
            "Expected ImageFolder layout with train/ and val/ directories. "
            f"Got data_root={data_root}. Prepare ImageNet first."
        )
    train_dataset = ImageFolder(str(train_root), transform=transform)
    val_dataset = ImageFolder(str(val_root), transform=transform)
    if train_dataset.class_to_idx != val_dataset.class_to_idx:
        raise ValueError("train and val class_to_idx differ; check ImageNet val organization.")
    print(
        f"[Data] train={len(train_dataset)}, official_val={len(val_dataset)}, "
        f"classes={len(train_dataset.classes)}"
    )
    return train_dataset, val_dataset


def encode_spatial_topk_features(
    backbone: nn.Module,
    sae: nn.Module,
    images: torch.Tensor,
    topk: int,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    out = backbone(images)
    spatial = out["spatial_features"]  # [B, P, 2048]
    bsz, spatial_dim, feat_dim = spatial.shape
    sparse_flat = sae.encode(spatial.reshape(-1, feat_dim))
    sparse = sparse_flat.reshape(bsz, spatial_dim, sae.hidden_dim)
    if topk <= 0 or topk > int(sae.hidden_dim):
        raise ValueError(f"spatial_topk must be in [1, hidden_dim], got {topk}.")
    values, concept_indices = torch.topk(sparse, k=int(topk), dim=2)
    return concept_indices.to(torch.int32), values, int(spatial_dim)


def cast_feature_dtype(features: torch.Tensor, dtype_name: str) -> torch.Tensor:
    if dtype_name == "float16":
        return features.to(torch.float16)
    if dtype_name == "bfloat16":
        return features.to(torch.bfloat16)
    if dtype_name == "float32":
        return features.to(torch.float32)
    raise ValueError(f"Unsupported feature dtype: {dtype_name}")


def save_sparse_spatial_shard(
    split_dir: Path,
    shard_id: int,
    concept_indices: torch.Tensor,
    concept_values: torch.Tensor,
    labels: torch.Tensor,
    indices: torch.Tensor,
    source_indices: Optional[torch.Tensor] = None,
) -> Dict[str, Any]:
    def compact(tensor: torch.Tensor, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        if dtype is not None:
            tensor = tensor.to(dtype)
        return tensor.detach().clone().contiguous()

    feature_name = f"shard_{shard_id:06d}.pt"
    index_name = f"shard_{shard_id:06d}.index.pt"
    torch.save(
        {
            "concept_indices": compact(concept_indices, torch.int32),
            "concept_values": compact(concept_values),
        },
        split_dir / feature_name,
    )
    index_payload = {
        "labels": compact(labels, torch.long),
        "indices": compact(indices, torch.long),
    }
    if source_indices is not None:
        index_payload["source_indices"] = compact(source_indices, torch.long)
    torch.save(index_payload, split_dir / index_name)
    shard_meta = {
        "feature_file": feature_name,
        "index_file": index_name,
        "count": int(labels.numel()),
        "first_index": int(indices[0].item()) if indices.numel() else None,
        "last_index": int(indices[-1].item()) if indices.numel() else None,
    }
    if source_indices is not None:
        shard_meta["source_first_index"] = int(source_indices[0].item()) if source_indices.numel() else None
        shard_meta["source_last_index"] = int(source_indices[-1].item()) if source_indices.numel() else None
    return shard_meta


def extract_split_to_shards(
    split_name: str,
    dataset: Dataset,
    class_to_idx: Dict[str, int],
    feature_root: Path,
    backbone: nn.Module,
    sae: nn.Module,
    device: torch.device,
    args: argparse.Namespace,
) -> None:
    split_dir = feature_root / split_name
    manifest_path = split_dir / "manifest.json"
    progress_path = split_dir / "progress.json"

    if manifest_path.exists() and not args.overwrite_features:
        print(f"[Features] {split_name}: manifest exists, skip extraction: {manifest_path}")
        return

    if args.overwrite_features and split_dir.exists():
        shutil.rmtree(split_dir)
    split_dir.mkdir(parents=True, exist_ok=True)

    start_index = 0
    shards: List[Dict[str, Any]] = []
    if progress_path.exists() and not args.overwrite_features:
        progress = read_json(progress_path)
        start_index = int(progress.get("next_index", 0))
        shards = list(progress.get("shards", []))
        print(f"[Features] {split_name}: resuming from dataset index {start_index}")

    indexed = IndexedDataset(dataset, start_index=start_index)
    loader = DataLoader(
        indexed,
        batch_size=args.batch_size_extract,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        persistent_workers=args.num_workers > 0,
    )

    backbone.eval()
    sae.eval()
    concept_index_buffers: List[torch.Tensor] = []
    concept_value_buffers: List[torch.Tensor] = []
    label_buffers: List[torch.Tensor] = []
    index_buffers: List[torch.Tensor] = []
    buffered = 0
    shard_id = len(shards)
    processed = start_index
    spatial_dim_seen: Optional[int] = None
    t0 = time.time()

    pbar = progress_bar(loader, desc=f"[Extract:{split_name}]")
    with torch.inference_mode():
        for images, labels, indices in pbar:
            images = images.to(device, non_blocking=True)
            if args.extract_amp and device.type == "cuda":
                amp_dtype = torch.bfloat16 if args.extract_amp_dtype == "bfloat16" else torch.float16
                with torch.autocast(device_type="cuda", dtype=amp_dtype):
                    concept_indices, concept_values, spatial_dim = encode_spatial_topk_features(
                        backbone,
                        sae,
                        images,
                        topk=args.spatial_topk,
                    )
            else:
                concept_indices, concept_values, spatial_dim = encode_spatial_topk_features(
                    backbone,
                    sae,
                    images,
                    topk=args.spatial_topk,
                )
            if spatial_dim_seen is None:
                spatial_dim_seen = int(spatial_dim)
            elif spatial_dim_seen != int(spatial_dim):
                raise RuntimeError(
                    f"Spatial dimension changed within split {split_name}: "
                    f"{spatial_dim_seen} vs {spatial_dim}"
                )

            concept_indices_cpu = concept_indices.detach().cpu().to(torch.int32)
            concept_values_cpu = cast_feature_dtype(concept_values.detach().cpu(), args.feature_dtype)
            labels_cpu = labels.detach().cpu().long()
            indices_cpu = indices.detach().cpu().long()

            concept_index_buffers.append(concept_indices_cpu)
            concept_value_buffers.append(concept_values_cpu)
            label_buffers.append(labels_cpu)
            index_buffers.append(indices_cpu)
            buffered += int(labels_cpu.numel())
            processed += int(labels_cpu.numel())

            while buffered >= args.shard_size:
                concept_indices_all = torch.cat(concept_index_buffers, dim=0)
                concept_values_all = torch.cat(concept_value_buffers, dim=0)
                labels_all = torch.cat(label_buffers, dim=0)
                indices_all = torch.cat(index_buffers, dim=0)

                shard_concept_indices = concept_indices_all[: args.shard_size]
                shard_concept_values = concept_values_all[: args.shard_size]
                shard_labels = labels_all[: args.shard_size]
                shard_indices = indices_all[: args.shard_size]
                shards.append(
                    save_sparse_spatial_shard(
                        split_dir,
                        shard_id,
                        shard_concept_indices,
                        shard_concept_values,
                        shard_labels,
                        shard_indices,
                    )
                )
                shard_id += 1

                concept_indices_all = concept_indices_all[args.shard_size :]
                concept_values_all = concept_values_all[args.shard_size :]
                labels_all = labels_all[args.shard_size :]
                indices_all = indices_all[args.shard_size :]
                concept_index_buffers = [concept_indices_all] if concept_indices_all.numel() else []
                concept_value_buffers = [concept_values_all] if concept_values_all.numel() else []
                label_buffers = [labels_all] if labels_all.numel() else []
                index_buffers = [indices_all] if indices_all.numel() else []
                buffered = int(labels_all.numel())

                write_json(
                    progress_path,
                    {
                        "split": split_name,
                        "next_index": processed - buffered,
                        "shards": shards,
                    },
                )

            pbar.set_postfix(done=processed, shards=shard_id)
            del images, concept_indices, concept_values

    if buffered > 0:
        concept_indices_all = torch.cat(concept_index_buffers, dim=0)
        concept_values_all = torch.cat(concept_value_buffers, dim=0)
        labels_all = torch.cat(label_buffers, dim=0)
        indices_all = torch.cat(index_buffers, dim=0)
        shards.append(
            save_sparse_spatial_shard(
                split_dir,
                shard_id,
                concept_indices_all,
                concept_values_all,
                labels_all,
                indices_all,
            )
        )
        shard_id += 1

    if spatial_dim_seen is None:
        raise RuntimeError(f"No features were extracted for split {split_name}.")
    spatial_side = int(round(float(spatial_dim_seen) ** 0.5))
    if spatial_side * spatial_side != int(spatial_dim_seen):
        raise ValueError(f"Expected square spatial grid, got spatial_dim={spatial_dim_seen}.")

    manifest = {
        "split": split_name,
        "num_samples": len(dataset),
        "feature_dim": int(args.sae_hidden_dim),
        "bottleneck": BOTTLENECK_NAME,
        "model": MODEL_NAME,
        "layout": FEATURE_LAYOUT,
        "cache_schema_version": 1,
        "stored_tensors": {
            "feature_file": ["concept_indices:int32[N,P,K]", "concept_values:dtype[N,P,K]"],
            "index_file": ["labels:int64[N]", "indices:int64[N]"],
        },
        "feature_dtype": args.feature_dtype,
        "shard_size": int(args.shard_size),
        "num_shards": len(shards),
        "shards": shards,
        "class_to_idx": class_to_idx,
        "classes": [cls for cls, _ in sorted(class_to_idx.items(), key=lambda kv: kv[1])],
        "sae_hidden_dim": int(args.sae_hidden_dim),
        "sae_k_sparse": int(args.sae_k_sparse),
        "spatial_topk": int(args.spatial_topk),
        "spatial_dim": int(spatial_dim_seen),
        "spatial_grid": [int(spatial_side), int(spatial_side)],
        "created_seconds": round(time.time() - t0, 3),
    }
    write_json(manifest_path, manifest)
    if progress_path.exists():
        progress_path.unlink()
    print(f"[Features] {split_name}: wrote {len(shards)} shards to {split_dir}")


def load_manifest(feature_root: Path, split: str) -> Dict[str, Any]:
    path = feature_root / split / "manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing feature manifest: {path}. Run --stage prepare first.")
    return read_json(path)


class FeatureShardStream:
    def __init__(self, feature_root: Path, split: str):
        self.feature_root = feature_root
        self.split = split
        self.split_dir = feature_root / split
        self.manifest = load_manifest(feature_root, split)
        self._validate_manifest()
        self.shards = self.manifest["shards"]
        self.num_samples = int(self.manifest["num_samples"])
        self.hidden_dim = int(self.manifest["sae_hidden_dim"])
        self.feature_dim = self.hidden_dim
        self.spatial_dim = int(self.manifest["spatial_dim"])
        self.spatial_topk = int(self.manifest["spatial_topk"])

    def _validate_manifest(self) -> None:
        errors: List[str] = []
        if self.manifest.get("bottleneck") != BOTTLENECK_NAME:
            errors.append(f"bottleneck={self.manifest.get('bottleneck')!r}, expected {BOTTLENECK_NAME!r}")
        if self.manifest.get("layout") != FEATURE_LAYOUT:
            errors.append(f"layout={self.manifest.get('layout')!r}, expected {FEATURE_LAYOUT!r}")
        hidden_dim = int(self.manifest.get("sae_hidden_dim", 0))
        feature_dim = int(self.manifest.get("feature_dim", hidden_dim))
        if hidden_dim <= 0:
            errors.append(f"sae_hidden_dim={hidden_dim}, expected positive")
        if feature_dim not in (hidden_dim, hidden_dim * int(self.manifest.get("encoder_out_channels", 1))):
            errors.append(
                f"feature_dim={feature_dim}, expected raw cache sae_hidden_dim={hidden_dim} "
                "or sae_hidden_dim * encoder_out_channels"
            )
        if errors:
            raise RuntimeError(
                f"Feature manifest for split {self.split!r} is not a {BOTTLENECK_NAME} cache:\n  "
                + "\n  ".join(errors)
            )

    def iter_batches(
        self,
        batch_size: int,
        allowed_mask: Optional[np.ndarray] = None,
        shuffle: bool = False,
        seed: int = 0,
    ) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        rng = np.random.default_rng(seed)
        order = np.arange(len(self.shards))
        if shuffle:
            rng.shuffle(order)

        for shard_pos in order:
            shard = self.shards[int(shard_pos)]
            feat_obj = torch_load(self.split_dir / shard["feature_file"], map_location="cpu")
            index_obj = torch_load(self.split_dir / shard["index_file"], map_location="cpu")
            concept_indices = feat_obj["concept_indices"]
            concept_values = feat_obj["concept_values"]
            labels = index_obj["labels"]
            indices = index_obj["indices"]

            if allowed_mask is not None:
                mask_np = allowed_mask[indices.numpy()]
                if not np.any(mask_np):
                    continue
                mask = torch.from_numpy(mask_np.astype(np.bool_))
                concept_indices = concept_indices[mask]
                concept_values = concept_values[mask]
                labels = labels[mask]

            n = int(labels.numel())
            if n == 0:
                continue
            if shuffle:
                perm = torch.from_numpy(rng.permutation(n))
                concept_indices = concept_indices[perm]
                concept_values = concept_values[perm]
                labels = labels[perm]

            for start in range(0, n, batch_size):
                end = min(start + batch_size, n)
                yield concept_indices[start:end], concept_values[start:end], labels[start:end]


class _OnlineIndexSubset(Dataset):
    def __init__(self, dataset: Dataset, indices: np.ndarray):
        self.dataset = dataset
        self.indices = np.asarray(indices, dtype=np.int64)

    def __len__(self) -> int:
        return int(self.indices.size)

    def __getitem__(self, position: int) -> Tuple[torch.Tensor, int, int]:
        original_index = int(self.indices[int(position)])
        item = self.dataset[original_index]
        if not isinstance(item, (tuple, list)) or len(item) < 2:
            raise RuntimeError("OnlineFeatureStream dataset items must contain at least (image, label).")
        image, label = item[0], item[1]
        return image, int(label), original_index


class OnlineFeatureStream:
    """Extract sparse SAE maps from images using the FeatureShardStream batch interface."""

    def __init__(
        self,
        dataset: Dataset,
        sample_indices: np.ndarray,
        classes: List[str],
        backbone: nn.Module,
        sae: nn.Module,
        device: torch.device,
        args: argparse.Namespace,
        *,
        split: str = "online_train",
    ):
        self.dataset = dataset
        self.sample_indices = np.asarray(sample_indices, dtype=np.int64)
        self.classes = [str(cls) for cls in classes]
        self.backbone = backbone
        self.sae = sae
        self.device = device
        self.args = args
        self.split = str(split)
        self.num_samples = int(self.sample_indices.size)
        self.hidden_dim = int(args.sae_hidden_dim)
        self.feature_dim = self.hidden_dim
        self.spatial_dim = int(args.spatial_dim)
        self.spatial_topk = int(args.spatial_topk)
        self.manifest: Dict[str, Any] = {
            "source": "online",
            "split": self.split,
            "bottleneck": BOTTLENECK_NAME,
            "layout": FEATURE_LAYOUT,
            "classes": self.classes,
            "num_samples": self.num_samples,
            "sae_hidden_dim": self.hidden_dim,
            "feature_dim": self.feature_dim,
            "spatial_dim": self.spatial_dim,
            "spatial_topk": self.spatial_topk,
        }

    def iter_batches(
        self,
        batch_size: int,
        allowed_mask: Optional[np.ndarray] = None,
        shuffle: bool = False,
        seed: int = 0,
    ) -> Iterator[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        indices = self.sample_indices
        if allowed_mask is not None:
            mask_np = allowed_mask[indices]
            indices = indices[mask_np.astype(np.bool_)]
        if indices.size == 0:
            return

        rng = np.random.default_rng(seed)
        ordered_indices = np.array(indices, copy=True)
        if shuffle:
            rng.shuffle(ordered_indices)

        subset = _OnlineIndexSubset(self.dataset, ordered_indices)
        generator = torch.Generator()
        generator.manual_seed(int(seed))
        loader = DataLoader(
            subset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=int(getattr(self.args, "num_workers", 0)),
            pin_memory=bool(getattr(self.args, "pin_memory", False)),
            persistent_workers=False,
            generator=generator,
        )

        self.backbone.eval()
        self.sae.eval()
        for images, labels, _original_indices in loader:
            images = images.to(self.device, non_blocking=True)
            with torch.inference_mode():
                if bool(getattr(self.args, "extract_amp", False)) and self.device.type == "cuda":
                    amp_dtype = torch.bfloat16 if self.args.extract_amp_dtype == "bfloat16" else torch.float16
                    with torch.autocast(device_type="cuda", dtype=amp_dtype):
                        concept_indices, concept_values, spatial_dim = encode_spatial_topk_features(
                            self.backbone,
                            self.sae,
                            images,
                            topk=self.spatial_topk,
                        )
                else:
                    concept_indices, concept_values, spatial_dim = encode_spatial_topk_features(
                        self.backbone,
                        self.sae,
                        images,
                        topk=self.spatial_topk,
                    )
                if int(spatial_dim) != self.spatial_dim:
                    raise RuntimeError(
                        f"Online spatial dimension changed for split {self.split}: "
                        f"{spatial_dim} vs expected {self.spatial_dim}"
                    )
                concept_indices = concept_indices.detach()
                concept_values = cast_feature_dtype(concept_values.detach(), str(self.args.feature_dtype))
            yield concept_indices, concept_values, labels.long()


def collect_labels_from_indices(feature_root: Path, split: str) -> np.ndarray:
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
        missing = int((~seen).sum())
        raise RuntimeError(f"Index files for {split} are incomplete; missing {missing} samples.")
    return labels


def make_allowed_mask(num_samples: int, indices: np.ndarray) -> np.ndarray:
    mask = np.zeros(num_samples, dtype=np.bool_)
    mask[np.asarray(indices, dtype=np.int64)] = True
    return mask


def parse_float_grid(value: str) -> List[float]:
    parts = [p.strip() for p in str(value).split(",") if p.strip()]
    if not parts:
        raise argparse.ArgumentTypeError("Grid must contain at least one value.")
    return [float(p) for p in parts]


def parse_float_grid_or_single(value: Optional[str], single_value: float) -> List[float]:
    if value is None or str(value).strip().lower() in ("", "none", "single"):
        return [float(single_value)]
    return parse_float_grid(value)


def parse_int_grid(value: str) -> List[int]:
    parts = [p.strip() for p in str(value).split(",") if p.strip()]
    if not parts:
        raise argparse.ArgumentTypeError("Grid must contain at least one value.")
    out = [int(p) for p in parts]
    if any(v <= 0 for v in out):
        raise argparse.ArgumentTypeError("Integer grid values must be positive.")
    return out


def parse_int_grid_or_single(value: Optional[str], single_value: int) -> List[int]:
    if value is None or str(value).strip().lower() in ("", "none", "single"):
        return [int(single_value)]
    return parse_int_grid(value)


def parse_seed_grid(value: str) -> List[int]:
    parts = [p.strip() for p in str(value).split(",") if p.strip()]
    if not parts:
        raise argparse.ArgumentTypeError("Seed grid must contain at least one value.")
    out = [int(p) for p in parts]
    if any(v < 0 for v in out):
        raise argparse.ArgumentTypeError("Seed grid values must be non-negative.")
    return out


def parse_seed_grid_or_single(value: Optional[str], single_value: int) -> List[int]:
    if value is None or str(value).strip().lower() in ("", "none", "single"):
        return [int(single_value)]
    return parse_seed_grid(value)


def parse_bool_grid(value: str) -> List[bool]:
    parts = [p.strip() for p in str(value).split(",") if p.strip()]
    if not parts:
        raise argparse.ArgumentTypeError("Boolean grid must contain at least one value.")
    return [str2bool(p) for p in parts]


def parse_bool_grid_or_single(value: Optional[str], single_value: bool) -> List[bool]:
    if value is None or str(value).strip().lower() in ("", "none", "single"):
        return [bool(single_value)]
    return parse_bool_grid(value)


def indices_sha256(indices: np.ndarray) -> str:
    arr = np.asarray(indices, dtype=np.int64)
    return hashlib.sha256(arr.tobytes()).hexdigest()


def verify_split_hashes(
    expected: Optional[Dict[str, str]],
    splits: Dict[str, np.ndarray],
    source: str,
) -> None:
    if not expected:
        return
    if not isinstance(expected, dict):
        raise ValueError(f"Expected split hashes from {source} must be a JSON object.")

    actual = {
        "search_train": indices_sha256(splits["search_train_indices"]),
        "search_val": indices_sha256(splits["search_val_indices"]),
        "final_holdout_val": indices_sha256(splits["final_holdout_val_indices"]),
        "final_train": indices_sha256(splits["final_train_indices"]),
    }
    unknown = sorted(set(expected) - set(actual))
    if unknown:
        raise ValueError(f"Unknown split hash names in {source}: " + ", ".join(unknown))
    mismatched = sorted(key for key, value in expected.items() if actual[key] != value)
    if mismatched:
        raise RuntimeError(
            f"Current split indices do not match {source} for: " + ", ".join(mismatched)
        )


def create_or_load_protocol_splits(feature_root: Path, args: argparse.Namespace) -> Dict[str, np.ndarray]:
    split_path = Path(args.split_file) if args.split_file else Path(args.output_dir) / "splits" / (
        "imagenet_protocol_"
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

    labels = collect_labels_from_indices(feature_root, "train")
    num_classes = int(labels.max()) + 1
    rng = np.random.default_rng(args.split_seed)
    search_train_parts: List[np.ndarray] = []
    search_val_parts: List[np.ndarray] = []
    final_holdout_parts: List[np.ndarray] = []

    needed_per_class = (
        int(args.search_train_per_class)
        + int(args.search_val_per_class)
        + int(args.final_val_per_class)
    )
    for class_id in range(num_classes):
        class_indices = np.flatnonzero(labels == class_id)
        if class_indices.size < needed_per_class:
            raise ValueError(
                f"Class {class_id} has only {class_indices.size} samples; "
                f"cannot take disjoint search_train={args.search_train_per_class}, "
                f"search_val={args.search_val_per_class}, "
                f"final_val={args.final_val_per_class}."
            )
        rng.shuffle(class_indices)
        start = 0
        end = start + int(args.search_train_per_class)
        search_train_parts.append(np.sort(class_indices[start:end]))
        start = end
        end = start + int(args.search_val_per_class)
        search_val_parts.append(np.sort(class_indices[start:end]))
        start = end
        end = start + int(args.final_val_per_class)
        final_holdout_parts.append(np.sort(class_indices[start:end]))

    search_train_indices = np.concatenate(search_train_parts).astype(np.int64)
    search_val_indices = np.concatenate(search_val_parts).astype(np.int64)
    final_holdout_val_indices = np.concatenate(final_holdout_parts).astype(np.int64)
    final_train_mask = np.ones(labels.shape[0], dtype=np.bool_)
    final_train_mask[final_holdout_val_indices] = False
    final_train_indices = np.flatnonzero(final_train_mask).astype(np.int64)

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
        f"[Split] wrote {split_path}: "
        f"search_train={search_train_indices.size}, "
        f"search_val={search_val_indices.size}, "
        f"final_holdout_val={final_holdout_val_indices.size}, "
        f"final_train={final_train_indices.size}"
    )
    return {
        "search_train_indices": search_train_indices,
        "search_val_indices": search_val_indices,
        "final_holdout_val_indices": final_holdout_val_indices,
        "final_train_indices": final_train_indices,
    }


def materialize_train_subset(
    feature_root: Path,
    subset_name: str,
    source_indices: np.ndarray,
    args: argparse.Namespace,
) -> None:
    source_stream = FeatureShardStream(feature_root, "train")
    subset_dir = feature_root / subset_name
    manifest_path = subset_dir / "manifest.json"
    source_indices = np.asarray(source_indices, dtype=np.int64)
    subset_hash = indices_sha256(source_indices)

    if manifest_path.exists() and not args.overwrite_subsets:
        manifest = read_json(manifest_path)
        if (
            int(manifest.get("num_samples", -1)) == int(source_indices.size)
            and manifest.get("source_indices_sha256") == subset_hash
        ):
            print(f"[Subset] {subset_name}: manifest exists, skip materialization: {manifest_path}")
            return
        raise RuntimeError(
            f"Existing subset manifest does not match requested indices: {manifest_path}. "
            "Use --overwrite_subsets true or a fresh --output_dir."
        )

    if subset_dir.exists():
        shutil.rmtree(subset_dir)
    subset_dir.mkdir(parents=True, exist_ok=True)

    selected_mask = make_allowed_mask(source_stream.num_samples, source_indices)
    local_lookup = np.full(source_stream.num_samples, -1, dtype=np.int64)
    local_lookup[source_indices] = np.arange(source_indices.size, dtype=np.int64)

    concept_index_buffers: List[torch.Tensor] = []
    concept_value_buffers: List[torch.Tensor] = []
    label_buffers: List[torch.Tensor] = []
    local_index_buffers: List[torch.Tensor] = []
    source_index_buffers: List[torch.Tensor] = []
    buffered = 0
    shard_id = 0
    shards: List[Dict[str, Any]] = []

    for shard in progress_bar(source_stream.shards, desc=f"[Subset:{subset_name}]"):
        feat_obj = torch_load(source_stream.split_dir / shard["feature_file"], map_location="cpu")
        index_obj = torch_load(source_stream.split_dir / shard["index_file"], map_location="cpu")
        concept_indices = feat_obj["concept_indices"]
        concept_values = feat_obj["concept_values"]
        labels = index_obj["labels"]
        indices = index_obj["indices"]
        mask_np = selected_mask[indices.numpy()]
        if not np.any(mask_np):
            continue

        mask = torch.from_numpy(mask_np.astype(np.bool_))
        subset_concept_indices = concept_indices[mask]
        subset_concept_values = concept_values[mask]
        subset_labels = labels[mask]
        subset_source_indices = indices[mask].long()
        subset_local_indices = torch.from_numpy(local_lookup[subset_source_indices.numpy()]).long()

        concept_index_buffers.append(subset_concept_indices)
        concept_value_buffers.append(subset_concept_values)
        label_buffers.append(subset_labels)
        local_index_buffers.append(subset_local_indices)
        source_index_buffers.append(subset_source_indices)
        buffered += int(subset_labels.numel())

        while buffered >= args.subset_shard_size:
            concept_indices_all = torch.cat(concept_index_buffers, dim=0)
            concept_values_all = torch.cat(concept_value_buffers, dim=0)
            labels_all = torch.cat(label_buffers, dim=0)
            local_indices_all = torch.cat(local_index_buffers, dim=0)
            source_indices_all = torch.cat(source_index_buffers, dim=0)

            shards.append(
                save_sparse_spatial_shard(
                    subset_dir,
                    shard_id,
                    concept_indices_all[: args.subset_shard_size],
                    concept_values_all[: args.subset_shard_size],
                    labels_all[: args.subset_shard_size],
                    local_indices_all[: args.subset_shard_size],
                    source_indices_all[: args.subset_shard_size],
                )
            )
            shard_id += 1

            concept_indices_all = concept_indices_all[args.subset_shard_size :]
            concept_values_all = concept_values_all[args.subset_shard_size :]
            labels_all = labels_all[args.subset_shard_size :]
            local_indices_all = local_indices_all[args.subset_shard_size :]
            source_indices_all = source_indices_all[args.subset_shard_size :]
            concept_index_buffers = [concept_indices_all] if concept_indices_all.numel() else []
            concept_value_buffers = [concept_values_all] if concept_values_all.numel() else []
            label_buffers = [labels_all] if labels_all.numel() else []
            local_index_buffers = [local_indices_all] if local_indices_all.numel() else []
            source_index_buffers = [source_indices_all] if source_indices_all.numel() else []
            buffered = int(labels_all.numel())

    if buffered > 0:
        concept_indices_all = torch.cat(concept_index_buffers, dim=0)
        concept_values_all = torch.cat(concept_value_buffers, dim=0)
        labels_all = torch.cat(label_buffers, dim=0)
        local_indices_all = torch.cat(local_index_buffers, dim=0)
        source_indices_all = torch.cat(source_index_buffers, dim=0)
        shards.append(
            save_sparse_spatial_shard(
                subset_dir,
                shard_id,
                concept_indices_all,
                concept_values_all,
                labels_all,
                local_indices_all,
                source_indices_all,
            )
        )

    written = int(sum(shard["count"] for shard in shards))
    if written != int(source_indices.size):
        raise RuntimeError(f"Subset {subset_name} wrote {written} samples, expected {source_indices.size}.")

    source_manifest = source_stream.manifest
    manifest = {
        "split": subset_name,
        "source_split": "train",
        "num_samples": int(source_indices.size),
        "source_num_samples": int(source_stream.num_samples),
        "source_indices_sha256": subset_hash,
        "feature_dim": int(source_manifest.get("sae_hidden_dim", args.sae_hidden_dim)),
        "bottleneck": source_manifest.get("bottleneck", BOTTLENECK_NAME),
        "model": source_manifest.get("model", MODEL_NAME),
        "layout": source_manifest.get("layout", FEATURE_LAYOUT),
        "cache_schema_version": int(source_manifest.get("cache_schema_version", 1)),
        "stored_tensors": source_manifest.get(
            "stored_tensors",
            {
                "feature_file": ["concept_indices:int32[N,P,K]", "concept_values:dtype[N,P,K]"],
                "index_file": ["labels:int64[N]", "indices:int64[N]"],
            },
        ),
        "feature_dtype": source_manifest.get("feature_dtype", "unknown"),
        "shard_size": int(args.subset_shard_size),
        "num_shards": len(shards),
        "shards": shards,
        "class_to_idx": source_manifest["class_to_idx"],
        "classes": source_manifest["classes"],
        "sae_hidden_dim": int(source_manifest.get("sae_hidden_dim", args.sae_hidden_dim)),
        "sae_k_sparse": int(source_manifest.get("sae_k_sparse", args.sae_k_sparse)),
        "spatial_topk": int(source_manifest.get("spatial_topk", args.spatial_topk)),
        "spatial_dim": int(source_manifest.get("spatial_dim", 49)),
        "spatial_grid": source_manifest.get("spatial_grid", [7, 7]),
    }
    write_json(manifest_path, manifest)
    print(f"[Subset] {subset_name}: wrote {len(shards)} shards to {subset_dir}")


def materialize_mixed_final_train(
    feature_root: Path,
    final_train_indices: np.ndarray,
    args: argparse.Namespace,
) -> None:
    """Mix cached training shards using the supplied final_train_indices."""

    source_stream = FeatureShardStream(feature_root, "train")
    subset_name = MIXED_FINAL_TRAIN_SPLIT
    subset_dir = feature_root / subset_name
    manifest_path = subset_dir / "manifest.json"
    source_indices = np.asarray(final_train_indices, dtype=np.int64)
    subset_hash = indices_sha256(source_indices)
    mix_seed = int(args.seed)
    mix_buffer_shards = int(FINAL_TRAIN_MIX_BUFFER_SHARDS)

    if manifest_path.exists():
        manifest = read_json(manifest_path)
        expected = {
            "num_samples": int(source_indices.size),
            "source_indices_sha256": subset_hash,
            "source_protocol_indices": "final_train_indices",
            "materialization": "mixed_shard_buffer",
            "mix_seed": mix_seed,
            "mix_buffer_shards": mix_buffer_shards,
        }
        mismatches = [
            f"{key}={manifest.get(key)!r}, expected {value!r}"
            for key, value in expected.items()
            if manifest.get(key) != value
        ]
        if mismatches:
            raise RuntimeError(
                f"Existing {subset_name} manifest does not match the requested final train stream:\n  "
                + "\n  ".join(mismatches)
                + f"\nRemove {subset_dir} manually if you intend to rebuild it."
            )
        print(f"[FinalTrainMixed] using existing {manifest_path}")
        return

    if subset_dir.exists():
        raise RuntimeError(
            f"{subset_dir} exists but has no manifest.json. Remove it manually before rebuilding."
        )

    subset_dir.mkdir(parents=True, exist_ok=False)
    print(
        f"[FinalTrainMixed] creating {subset_dir} from protocol final_train_indices: "
        f"samples={source_indices.size}, mix_seed={mix_seed}, buffer_shards={mix_buffer_shards}",
        flush=True,
    )

    selected_mask = make_allowed_mask(source_stream.num_samples, source_indices)
    local_lookup = np.full(source_stream.num_samples, -1, dtype=np.int64)
    local_lookup[source_indices] = np.arange(source_indices.size, dtype=np.int64)
    rng = np.random.default_rng(mix_seed)
    shard_order = np.arange(len(source_stream.shards))
    rng.shuffle(shard_order)

    shards: List[Dict[str, Any]] = []
    shard_id = 0
    written = 0

    def write_mixed_buffer(
        concept_index_buffers: List[torch.Tensor],
        concept_value_buffers: List[torch.Tensor],
        label_buffers: List[torch.Tensor],
        local_index_buffers: List[torch.Tensor],
        source_index_buffers: List[torch.Tensor],
    ) -> None:
        nonlocal shard_id, written
        if not label_buffers:
            return
        concept_indices_all = torch.cat(concept_index_buffers, dim=0)
        concept_values_all = torch.cat(concept_value_buffers, dim=0)
        labels_all = torch.cat(label_buffers, dim=0)
        local_indices_all = torch.cat(local_index_buffers, dim=0)
        source_indices_all = torch.cat(source_index_buffers, dim=0)
        n = int(labels_all.numel())
        if n == 0:
            return

        perm = torch.from_numpy(rng.permutation(n)).long()
        concept_indices_all = concept_indices_all[perm]
        concept_values_all = concept_values_all[perm]
        labels_all = labels_all[perm]
        local_indices_all = local_indices_all[perm]
        source_indices_all = source_indices_all[perm]

        for start in range(0, n, int(args.subset_shard_size)):
            end = min(start + int(args.subset_shard_size), n)
            shards.append(
                save_sparse_spatial_shard(
                    subset_dir,
                    shard_id,
                    concept_indices_all[start:end],
                    concept_values_all[start:end],
                    labels_all[start:end],
                    local_indices_all[start:end],
                    source_indices_all[start:end],
                )
            )
            shard_id += 1
            written += int(end - start)

    for start in progress_bar(
        range(0, len(shard_order), mix_buffer_shards),
        desc="[FinalTrainMixed]",
    ):
        concept_index_buffers: List[torch.Tensor] = []
        concept_value_buffers: List[torch.Tensor] = []
        label_buffers: List[torch.Tensor] = []
        local_index_buffers: List[torch.Tensor] = []
        source_index_buffers: List[torch.Tensor] = []

        for shard_pos in shard_order[start : start + mix_buffer_shards]:
            shard = source_stream.shards[int(shard_pos)]
            feat_obj = torch_load(source_stream.split_dir / shard["feature_file"], map_location="cpu")
            index_obj = torch_load(source_stream.split_dir / shard["index_file"], map_location="cpu")
            indices = index_obj["indices"]
            mask_np = selected_mask[indices.numpy()]
            if not np.any(mask_np):
                continue
            mask = torch.from_numpy(mask_np.astype(np.bool_))
            source_indices_selected = indices[mask].long()

            concept_index_buffers.append(feat_obj["concept_indices"][mask])
            concept_value_buffers.append(feat_obj["concept_values"][mask])
            label_buffers.append(index_obj["labels"][mask])
            local_index_buffers.append(torch.from_numpy(local_lookup[source_indices_selected.numpy()]).long())
            source_index_buffers.append(source_indices_selected)

        write_mixed_buffer(
            concept_index_buffers,
            concept_value_buffers,
            label_buffers,
            local_index_buffers,
            source_index_buffers,
        )

    if written != int(source_indices.size):
        raise RuntimeError(f"{subset_name} wrote {written} samples, expected {source_indices.size}.")

    source_manifest = source_stream.manifest
    manifest = {
        "split": subset_name,
        "source_split": "train",
        "source_protocol_indices": "final_train_indices",
        "materialization": "mixed_shard_buffer",
        "num_samples": int(source_indices.size),
        "source_num_samples": int(source_stream.num_samples),
        "source_indices_sha256": subset_hash,
        "mix_seed": mix_seed,
        "mix_buffer_shards": mix_buffer_shards,
        "feature_dim": int(source_manifest.get("sae_hidden_dim", args.sae_hidden_dim)),
        "bottleneck": source_manifest.get("bottleneck", BOTTLENECK_NAME),
        "model": source_manifest.get("model", MODEL_NAME),
        "layout": source_manifest.get("layout", FEATURE_LAYOUT),
        "cache_schema_version": int(source_manifest.get("cache_schema_version", 1)),
        "stored_tensors": source_manifest.get(
            "stored_tensors",
            {
                "feature_file": ["concept_indices:int32[N,P,K]", "concept_values:dtype[N,P,K]"],
                "index_file": ["labels:int64[N]", "indices:int64[N]"],
            },
        ),
        "feature_dtype": source_manifest.get("feature_dtype", "unknown"),
        "shard_size": int(args.subset_shard_size),
        "num_shards": len(shards),
        "shards": shards,
        "class_to_idx": source_manifest["class_to_idx"],
        "classes": source_manifest["classes"],
        "sae_hidden_dim": int(source_manifest.get("sae_hidden_dim", args.sae_hidden_dim)),
        "sae_k_sparse": int(source_manifest.get("sae_k_sparse", args.sae_k_sparse)),
        "spatial_topk": int(source_manifest.get("spatial_topk", args.spatial_topk)),
        "spatial_dim": int(source_manifest.get("spatial_dim", 49)),
        "spatial_grid": source_manifest.get("spatial_grid", [7, 7]),
    }
    write_json(manifest_path, manifest)
    print(f"[FinalTrainMixed] wrote {len(shards)} shards to {subset_dir}")


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


def build_classifier(args: argparse.Namespace, input_dim: int, num_classes: int, device: torch.device) -> nn.Module:
    expected_dim = int(args.sae_hidden_dim)
    if int(input_dim) != expected_dim:
        raise ValueError(f"input_dim={input_dim}, expected raw cache hidden_dim={expected_dim}.")
    classifier_mode = str(getattr(args, "classifier_mode", "spatial"))
    if classifier_mode == "spatial":
        model = SpatialConceptClassifier(
            hidden_dim=int(args.sae_hidden_dim),
            spatial_dim=int(args.spatial_dim),
            encoder_hidden_channels=int(args.encoder_hidden_channels),
            encoder_out_channels=int(args.encoder_out_channels),
            encoder_kernel_size=int(args.encoder_kernel_size),
            num_classes=num_classes,
            active_chunk_size=int(args.active_map_chunk_size),
        ).to(device)
        model.train_microbatch_size = int(args.encoder_microbatch_size)
        model.eval_microbatch_size = int(args.encoder_microbatch_size)
        return model
    if classifier_mode == "mc":
        model = MCStatsClassifier(
            hidden_dim=int(args.sae_hidden_dim),
            num_classes=num_classes,
            epsilon=float(getattr(args, "mc_channel_epsilon", 1e-4)),
            sqrt_count=bool(getattr(args, "mc_sqrt_count", True)),
            blocknorm=str(getattr(args, "mc_blocknorm", "stat")),
        ).to(device)
        model.train_microbatch_size = int(getattr(args, "encoder_microbatch_size", args.batch_size_head))
        model.eval_microbatch_size = int(getattr(args, "encoder_microbatch_size", args.batch_size_head))
        return model
    raise ValueError(f"Unsupported classifier_mode: {classifier_mode!r}")


def build_optimizer(args: argparse.Namespace, parameters: Any) -> optim.Optimizer:
    if args.optimizer == "adamw":
        return optim.AdamW(parameters, lr=args.lr, weight_decay=args.weight_decay)
    if args.optimizer == "adam":
        return optim.Adam(parameters, lr=args.lr, weight_decay=args.weight_decay)
    if args.optimizer == "sgd":
        return optim.SGD(
            parameters,
            lr=args.lr,
            momentum=args.sgd_momentum,
            weight_decay=args.weight_decay,
            nesterov=args.sgd_nesterov,
        )
    raise ValueError(f"Unsupported optimizer: {args.optimizer}")


def evaluate_stream(
    classifier: nn.Module,
    stream: FeatureShardStream,
    allowed_mask: Optional[np.ndarray],
    batch_size: int,
    device: torch.device,
) -> float:
    classifier.eval()
    correct = 0
    total = 0
    with torch.inference_mode():
        for concept_indices, concept_values, labels in stream.iter_batches(
            batch_size=batch_size,
            allowed_mask=allowed_mask,
            shuffle=False,
        ):
            labels = labels.to(device, non_blocking=True)
            batch_n = int(labels.numel())
            micro = max(int(getattr(classifier, "eval_microbatch_size", batch_size)), 1)
            for start in range(0, batch_n, micro):
                end = min(start + micro, batch_n)
                idx_mb = concept_indices[start:end].to(device, non_blocking=True)
                val_mb = concept_values[start:end].to(device, non_blocking=True)
                labels_mb = labels[start:end]
                logits = classifier(idx_mb, val_mb)
                pred = logits.argmax(dim=1)
                total += int(labels_mb.numel())
                correct += int(pred.eq(labels_mb).sum().item())
    return 100.0 * correct / max(total, 1)


def train_with_validation(
    classifier: nn.Module,
    train_stream: FeatureShardStream,
    val_stream: FeatureShardStream,
    train_mask: Optional[np.ndarray],
    val_mask: Optional[np.ndarray],
    args: argparse.Namespace,
    device: torch.device,
    checkpoint_path: Optional[Path],
    epochs: int,
    desc_prefix: str,
    scheduler_t_max: Optional[int] = None,
) -> Tuple[float, int, List[float], List[float]]:
    fit_normalizer = getattr(classifier, "fit_normalizer", None)
    if callable(fit_normalizer):
        fit_normalizer(
            train_stream,
            train_mask,
            batch_size=args.batch_size_head,
            device=device,
        )
    optimizer = build_optimizer(args, classifier.parameters())
    criterion = nn.CrossEntropyLoss()
    effective_t_max = int(scheduler_t_max) if scheduler_t_max is not None else int(epochs)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(effective_t_max, 1))

    best_acc = -1.0
    best_epoch = -1
    train_losses: List[float] = []
    val_accs: List[float] = []

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
        for concept_indices, concept_values, labels in progress_bar(
            batches,
            desc=f"[{desc_prefix}] epoch {epoch + 1}/{epochs}",
        ):
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
        val_acc = evaluate_stream(
            classifier,
            val_stream,
            allowed_mask=val_mask,
            batch_size=args.batch_size_head,
            device=device,
        )
        train_losses.append(train_loss)
        val_accs.append(val_acc)

        if val_acc > best_acc:
            best_acc = val_acc
            best_epoch = epoch
            if checkpoint_path is not None:
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": classifier.state_dict(),
                        "val_acc": val_acc,
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
                        },
                        "args": vars(args),
                    },
                    checkpoint_path,
                )

        scheduler.step()
        lr = scheduler.get_last_lr()[0]
        print(
            f"[{desc_prefix} epoch {epoch + 1}/{epochs}] train_loss={train_loss:.4f} "
            f"train_acc={train_acc:.2f}% val_acc={val_acc:.2f}% "
            f"best={best_acc:.2f}% lr={lr:.2e}",
            flush=True,
        )

    return best_acc, best_epoch, train_losses, val_accs


def run_prepare(args: argparse.Namespace) -> None:
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")
    backbone = create_resnet_backbone(args.use_clip, args.clip_variant, args.clip_pretrained).to(device)
    transform = build_imagenet_transform(backbone, args.use_clip)
    train_dataset, val_dataset = load_imagenet_folders(Path(args.data_root), transform)

    sae = create_image_sae(args.sae_hidden_dim, args.sae_k_sparse).to(device)
    load_image_sae_weights(Path(args.sae_checkpoint), sae, device)
    sae.eval()

    feature_root = Path(args.feature_dir)
    extract_split_to_shards(
        "train",
        train_dataset,
        train_dataset.class_to_idx,
        feature_root,
        backbone,
        sae,
        device,
        args,
    )
    extract_split_to_shards(
        "official_val",
        val_dataset,
        val_dataset.class_to_idx,
        feature_root,
        backbone,
        sae,
        device,
        args,
    )


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
        "protocol": "subset_hparam_search",
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


def verify_selection_splits(selection: Dict[str, Any], splits: Dict[str, np.ndarray]) -> None:
    verify_split_hashes(selection.get("split_indices_sha256"), splits, "selection.json")


def run_final(args: argparse.Namespace) -> None:
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

    materialize_mixed_final_train(feature_root, splits["final_train_indices"], args)
    train_stream = FeatureShardStream(feature_root, MIXED_FINAL_TRAIN_SPLIT)
    final_holdout_val_stream = FeatureShardStream(feature_root, "final_holdout_val")
    official_val_stream = FeatureShardStream(feature_root, "official_val")
    args.spatial_dim = int(train_stream.spatial_dim)

    num_classes = len(train_stream.manifest["classes"])
    classifier = build_classifier(args, train_stream.feature_dim, num_classes, device)
    final_ckpt = Path(args.output_dir) / "checkpoints" / "final_best.pth"
    final_scheduler_t_max = int(args.final_scheduler_t_max)
    print(
        f"[Final] epochs={args.final_epochs}, "
        f"scheduler_t_max={final_scheduler_t_max}, "
        f"train={train_stream.num_samples} ({MIXED_FINAL_TRAIN_SPLIT}), "
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
        scheduler_t_max=final_scheduler_t_max,
    )

    best_payload = torch_load(final_ckpt, map_location=device)
    classifier.load_state_dict(best_payload["model_state_dict"])
    official_val_as_test_acc = evaluate_stream(
        classifier,
        official_val_stream,
        allowed_mask=None,
        batch_size=args.batch_size_head,
        device=device,
    )
    print(f"[Final] official_val_as_test_acc={official_val_as_test_acc:.2f}%")

    result = {
        "dataset": "imagenet1k",
        "validation_setup": "subset_hparam_search__train_holdout_checkpoint_selection__official_val_as_test",
        "best_final_holdout_val_acc": best_holdout_acc,
        "best_final_epoch": int(best_epoch),
        "official_val_as_test_acc": official_val_as_test_acc,
        "final_epochs": int(args.final_epochs),
        "final_scheduler_t_max": int(final_scheduler_t_max),
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
            "official_val_as_test": int(official_val_stream.num_samples),
        },
        "feature_dir": str(feature_root),
        "args": vars(args),
    }
    result_path = Path(args.output_dir) / "imagenet_sae_results.json"
    write_json(result_path, result)
    print(f"[Final] wrote {result_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ImageNet-1K SAE bottleneck downstream classifier")

    parser.add_argument(
        "--stage",
        type=str,
        default="prepare",
        choices=["prepare", "select", "final", "all"],
    )
    parser.add_argument("--data_root", type=str, default=None)
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

    parser.add_argument("--batch_size_extract", type=int, default=256)
    parser.add_argument("--batch_size_head", type=int, default=2048)
    parser.add_argument("--shard_size", type=int, default=4096)
    parser.add_argument("--feature_dtype", type=str, default="float32", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--pin_memory", type=str2bool, default=True)
    parser.add_argument("--overwrite_features", type=str2bool, default=False)
    parser.add_argument("--subset_shard_size", type=int, default=4096)
    parser.add_argument("--overwrite_subsets", type=str2bool, default=False)

    parser.add_argument("--extract_amp", type=str2bool, default=False)
    parser.add_argument("--extract_amp_dtype", type=str, default="float16", choices=["float16", "bfloat16"])
    parser.add_argument("--channel_epsilon", type=float, default=1e-8, help=argparse.SUPPRESS)
    parser.add_argument("--spatial_topk", type=int, default=64)
    parser.add_argument("--spatial_dim", type=int, default=49, help=argparse.SUPPRESS)
    parser.add_argument("--classifier_mode", type=str, default="spatial", choices=["spatial", "mc"])
    parser.add_argument("--encoder_hidden_channels", type=int, default=8)
    parser.add_argument("--encoder_out_channels", type=int, default=2)
    parser.add_argument("--encoder_kernel_size", type=int, default=3)
    parser.add_argument(
        "--encoder_microbatch_size",
        type=int,
        default=128,
        help="Number of images processed per encoder microbatch inside one optimizer step.",
    )
    parser.add_argument(
        "--active_map_chunk_size",
        type=int,
        default=8192,
        help="Number of active per-concept maps encoded per internal chunk.",
    )
    parser.add_argument("--mc_channel_epsilon", type=float, default=1e-4)
    parser.add_argument("--mc_sqrt_count", type=str2bool, default=True)
    parser.add_argument("--mc_blocknorm", type=str, default="stat", choices=["none", "stat"])

    parser.add_argument("--search_train_per_class", type=int, default=100)
    parser.add_argument("--search_val_per_class", type=int, default=10)
    parser.add_argument("--final_val_per_class", type=int, default=50)
    parser.add_argument("--overwrite_split", type=str2bool, default=False)
    parser.add_argument("--split_seed", type=int, default=None)
    parser.add_argument("--expected_split_hashes", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--seed_grid",
        type=str,
        default=None,
        help=(
            "Comma-separated training seed candidates for select. If omitted, select uses "
            "--seed as a single point. The split is controlled independently by "
            "--split_file or --split_seed."
        ),
    )

    parser.add_argument("--search_epochs", type=int, default=50)
    parser.add_argument("--final_epochs", type=int, default=100)
    parser.add_argument("--final_scheduler_t_max", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--optimizer", type=str, default="sgd", choices=["adamw", "adam", "sgd"])
    parser.add_argument("--sgd_momentum", type=float, default=0.9)
    parser.add_argument("--sgd_nesterov", type=str2bool, default=False)
    parser.add_argument(
        "--sgd_momentum_grid",
        type=str,
        default=None,
        help=(
            "Comma-separated SGD momentum candidates. If omitted, select uses "
            "--sgd_momentum as a single point. Example: 0.9,0.95."
        ),
    )
    parser.add_argument(
        "--sgd_nesterov_grid",
        type=str,
        default=None,
        help=(
            "Comma-separated SGD nesterov candidates. If omitted, select uses "
            "--sgd_nesterov as a single point. Example: false,true."
        ),
    )
    parser.add_argument(
        "--lr_grid",
        type=str,
        default=None,
        help=(
            "Comma-separated LR candidates. If omitted, select uses --lr as a single point. "
            "Example: 0.005,0.01,0.02,0.03."
        ),
    )
    parser.add_argument(
        "--weight_decay_grid",
        type=str,
        default=None,
        help=(
            "Comma-separated weight decay candidates. If omitted, select uses --weight_decay "
            "as a single point. Example: 0,0.0001,0.001."
        ),
    )
    parser.add_argument(
        "--batch_size_head_grid",
        type=str,
        default=None,
        help=(
            "Comma-separated classifier batch-size candidates. If omitted, select uses "
            "--batch_size_head as a single point. Example: 512,1024,2048."
        ),
    )
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--save_select_checkpoints", type=str2bool, default=False)

    parser.add_argument("--device", type=str, default="cuda")

    args = parse_args_with_config(parser)
    missing_paths = [
        name
        for name in ("data_root", "output_dir", "sae_checkpoint")
        if not str(getattr(args, name, "") or "").strip()
    ]
    if missing_paths:
        raise ValueError(
            "The following paths must be provided through --config or command-line "
            "arguments: " + ", ".join(missing_paths)
        )
    if args.epochs is not None:
        args.search_epochs = int(args.epochs)
        args.final_epochs = int(args.epochs)
    if args.split_seed is None:
        args.split_seed = int(args.seed)
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
    if args.search_train_per_class <= 0 or args.search_val_per_class <= 0 or args.final_val_per_class <= 0:
        raise ValueError("Per-class split sizes must be positive.")
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


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    print(json.dumps(vars(args), indent=2), flush=True)

    if args.stage in ("prepare", "all"):
        run_prepare(args)
    if args.stage in ("select", "all"):
        run_select(args)
    if args.stage in ("final", "all"):
        run_final(args)
