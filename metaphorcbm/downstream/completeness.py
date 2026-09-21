
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Few-shot completeness evaluation for sparse concept bottlenecks.

The experiment compares a linear probe on pooled backbone features with a
linear head on sparse concept bottleneck features across supported datasets.
"""

from __future__ import annotations

import argparse
import json
import math
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset, TensorDataset

import torchvision
import torchvision.transforms as transforms
from torchvision import datasets as tv_datasets
from PIL import Image

from metaphorcbm.config import parse_args_with_config

from metaphorcbm.models import ImageSAE as FrameworkImageSAE
from metaphorcbm.models import ResNetWithHooks, create_clip_resnet_backbone
from .datasets import TinyImageNetDataset


# -----------------------------
# Utilities
# -----------------------------
def str2bool(v):
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


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Use deterministic cuDNN algorithms.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def l2_normalize(x: torch.Tensor, dim: int = 1, eps: float = 1e-12) -> torch.Tensor:
    return x / (x.norm(dim=dim, keepdim=True) + eps)


def to_dtype(x: torch.Tensor, dtype: str) -> torch.Tensor:
    if dtype == "float32":
        return x.float()
    if dtype == "float16":
        return x.half()
    if dtype == "bfloat16":
        return x.bfloat16()
    raise ValueError(f"Unsupported dtype: {dtype}")


def _stable_tag(s: str, n: int = 8) -> str:
    """Stable short tag for cache keys."""
    if s is None:
        return "none"
    ss = str(s)
    if ss.strip() == "":
        return "none"
    return hashlib.md5(ss.encode("utf-8")).hexdigest()[:n]


# -----------------------------
# Backbone and SAE loading
# -----------------------------
def create_resnet_backbone(
    use_clip: bool = True,
    clip_variant: str = "RN50",
    clip_pretrained: str = "openai",
):
    """
    Create a backbone exposing spatial, pooled, and raw features.
    For layer4 features from a 224 x 224 input:
      - 'spatial_features': [B, 49, 2048]
      - 'pooled_features' : [B, 2048]
      - 'raw_features'    : [B, 2048, 7, 7]
    """
    if use_clip:
        print(f"[Backbone] Using CLIP-{clip_variant} visual encoder with hooks")
        cfg = {
            "clip_variant": clip_variant,
            "pretrained": clip_pretrained,
            "hook_layer": "layer4",
            "freeze_backbone": True,
            "return_features": False,
        }
        backbone = create_clip_resnet_backbone(cfg)
    else:
        print("[Backbone] Using torchvision ResNet-50 with hooks (ImageNet1K_V2)")
        backbone = ResNetWithHooks(
            weights="IMAGENET1K_V2",
            hook_layer="layer4",
            freeze_backbone=True,
            return_features=False,
        )

    backbone.eval()
    return backbone


def create_image_sae(hidden_dim: int = 8192, k_sparse: int = 16):
    sae = FrameworkImageSAE(
        input_dim=2048,
        hidden_dim=hidden_dim,
        k_sparse=k_sparse,
        use_bias=False,
        activation="relu",
        initialization="fan_in",
    )
    sae.eval()
    return sae


def load_image_sae_weights(checkpoint_path: str, image_sae: nn.Module, map_location="cpu") -> None:
    """Load image SAE weights from nested, prefixed, or direct state dictionaries."""
    if checkpoint_path is None or str(checkpoint_path).strip() == "":
        raise ValueError("An ImageSAE checkpoint must be provided.")

    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"SAE checkpoint not found: {ckpt_path}")

    print(f"[SAE] Loading checkpoint: {ckpt_path}")
    checkpoint = torch.load(str(ckpt_path), map_location=map_location, weights_only=False)

    # Common formats:
    # 1) {'model_state_dict': {'image_sae': state_dict, ...}, ...}
    # 2) {'model_state_dict': state_dict}
    # 3) {'image_sae': state_dict}
    # 4) state_dict directly
    state_dict = None

    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            msd = checkpoint["model_state_dict"]
            if isinstance(msd, dict) and "image_sae" in msd and isinstance(msd["image_sae"], dict):
                state_dict = msd["image_sae"]
                print("[SAE] Detected format: checkpoint['model_state_dict']['image_sae']")
            else:
                state_dict = msd
                print("[SAE] Detected format: checkpoint['model_state_dict']")
        elif "image_sae" in checkpoint and isinstance(checkpoint["image_sae"], dict):
            state_dict = checkpoint["image_sae"]
            print("[SAE] Detected format: checkpoint['image_sae']")
        else:
            state_dict = checkpoint

    if state_dict is None or not isinstance(state_dict, dict):
        raise RuntimeError(f"Unsupported ImageSAE checkpoint format: {ckpt_path}")

    # If keys are prefixed with 'image_sae.', strip it.
    if any(k.startswith("image_sae.") for k in state_dict.keys()):
        filtered = {}
        for k, v in state_dict.items():
            if k.startswith("image_sae."):
                filtered[k[len("image_sae.") :]] = v
        state_dict = filtered
        print("[SAE] Stripped 'image_sae.' prefix from keys.")

    image_sae.load_state_dict(state_dict, strict=True)
    print("[SAE] Loaded weights successfully.")


# -----------------------------
# Bottleneck construction from maximum activations and activation counts
# -----------------------------
def compute_channel_statistics(spatial_activations: torch.Tensor, epsilon: float = 1e-6) -> torch.Tensor:
    """
    Args:
        spatial_activations: [B, P, C] sparse activations across P spatial positions (P=49)
        epsilon: threshold for "activated" in count statistic

    Returns:
        features: [B, 2*C] in interleaved order: [m1, c1, m2, c2, ...]
    """
    B, P, C = spatial_activations.shape

    # m_j: max intensity over spatial positions
    max_intensities = spatial_activations.max(dim=1)[0]  # [B, C]

    # c_j: activated area count (#positions with activation > epsilon)
    area_counts = (spatial_activations > epsilon).float().sum(dim=1)  # [B, C]

    # Interleave per concept: [m_j, c_j]
    stacked = torch.stack([max_intensities, area_counts], dim=2)  # [B, C, 2]
    feats = stacked.view(B, 2 * C)  # [B, 2C]

    return feats


# -----------------------------
# Block normalization for m/c features
# -----------------------------
class BlockNormalizer:
    """
    Block normalization for [B, 2C] features:
      - mode='stat': global μ/σ per block (m vs c), shape [2]
      - mode='per_concept': μ/σ per concept per block, shape [C,2]
      - apply_sqrt_on_counts: optionally apply sqrt to c-block before z-score
    """

    def __init__(self, mode: str = "stat", eps: float = 1e-8, apply_sqrt_on_counts: bool = True):
        assert mode in ("stat", "per_concept")
        self.mode = mode
        self.eps = eps
        self.apply_sqrt_on_counts = apply_sqrt_on_counts
        self.fitted = False
        self.mu: Optional[torch.Tensor] = None
        self.sigma: Optional[torch.Tensor] = None

    @torch.no_grad()
    def fit_from_loader(
        self,
        train_loader: DataLoader,
        backbone: nn.Module,
        sae: nn.Module,
        epsilon: float,
        device: torch.device,
        max_samples: Optional[int] = None,
    ) -> "BlockNormalizer":
        """Estimate feature means and scales from up to max_samples training images."""
        backbone.eval()
        sae.eval()

        sum_b = None
        sqsum_b = None
        N = 0

        sum_pc = None
        sqsum_pc = None
        N_per = 0

        seen = 0

        for images, _ in tqdm(train_loader, desc="[Norm] Fitting BlockNormalizer", leave=False):
            images = images.to(device, non_blocking=True)
            out = backbone(images)
            sp = out["spatial_features"]  # [B,49,2048]
            s = sae(sp)["sparse_activations"]  # [B,49,C]
            feats = compute_channel_statistics(s, epsilon=epsilon)  # [B,2C]

            B, D = feats.shape
            C = D // 2
            X = feats.view(B, C, 2).float()

            if self.apply_sqrt_on_counts:
                X[:, :, 1] = torch.sqrt(torch.clamp(X[:, :, 1], min=0.0))

            if self.mode == "stat":
                if sum_b is None:
                    sum_b = torch.zeros(2, dtype=torch.float64)
                    sqsum_b = torch.zeros(2, dtype=torch.float64)
                sum_b += X.sum(dim=(0, 1)).double().cpu()
                sqsum_b += (X * X).sum(dim=(0, 1)).double().cpu()
                N += (B * C)
            else:
                if sum_pc is None:
                    sum_pc = torch.zeros(C, 2, dtype=torch.float64)
                    sqsum_pc = torch.zeros(C, 2, dtype=torch.float64)
                sum_pc += X.sum(dim=0).double().cpu()
                sqsum_pc += (X * X).sum(dim=0).double().cpu()
                N_per += B

            seen += B
            if max_samples is not None and seen >= max_samples:
                break

        if self.mode == "stat":
            mu = (sum_b / max(N, 1)).to(torch.float32)
            var = (sqsum_b / max(N, 1)) - (mu.double() ** 2)
            sigma = torch.sqrt(torch.clamp(var, min=1e-12)).to(torch.float32)
        else:
            mu = (sum_pc / max(N_per, 1)).to(torch.float32)
            var = (sqsum_pc / max(N_per, 1)) - (mu.double() ** 2)
            sigma = torch.sqrt(torch.clamp(var, min=1e-12)).to(torch.float32)

        self.mu = mu
        self.sigma = torch.clamp(sigma, min=self.eps)
        self.fitted = True
        return self

    def transform(self, features: torch.Tensor) -> torch.Tensor:
        """
        Return float32 z-scores per block or per concept for features [B, 2C].
        """
        assert self.fitted, "BlockNormalizer not fitted. Call fit_from_loader first."
        B, D = features.shape
        C = D // 2
        X = features.view(B, C, 2).float()

        if self.apply_sqrt_on_counts:
            X[:, :, 1] = torch.sqrt(torch.clamp(X[:, :, 1], min=0.0))

        if self.mode == "stat":
            mu = self.mu.view(1, 1, 2).to(X.device)
            sg = self.sigma.view(1, 1, 2).to(X.device)
        else:
            mu = self.mu.view(1, C, 2).to(X.device)
            sg = self.sigma.view(1, C, 2).to(X.device)

        X = (X - mu) / sg
        return X.view(B, D)


class COCOClassificationDataset(torch.utils.data.Dataset):
    """
    Assign each COCO image the category with the largest total annotated area.

    Categories come from instances_train2017.json or instances_val2017.json.
    Optional filtering retains only images annotated with a single category.
    """

    def __init__(
        self,
        root: str,
        train: bool = True,
        transform=None,
        single_label_only: bool = False,
        max_samples_per_class: Optional[int] = None,
    ):
        """
        Args:
            root: Path to COCO data directory (should contain train2017/, val2017/, annotations/)
            train: If True, use train2017; else use val2017
            transform: Image transforms
            single_label_only: If True, only keep images with exactly one category
            max_samples_per_class: Limit samples per class (for balancing)
        """
        self.root = Path(root)
        self.train = train
        self.transform = transform
        self.single_label_only = single_label_only
        self.max_samples_per_class = max_samples_per_class

        split = "train" if train else "val"
        self.image_dir = self.root / f"{split}2017"
        ann_file = self.root / "annotations" / f"instances_{split}2017.json"

        if not self.image_dir.exists():
            raise FileNotFoundError(f"COCO image directory not found: {self.image_dir}")
        if not ann_file.exists():
            raise FileNotFoundError(f"COCO annotation file not found: {ann_file}")

        print(f"[COCO] Loading annotations: {ann_file}")
        with open(ann_file, "r", encoding="utf-8") as f:
            coco_data = json.load(f)

        # Map noncontiguous category IDs to consecutive class indices.
        categories = sorted(coco_data["categories"], key=lambda x: x["id"])
        self.cat_id_to_idx = {cat["id"]: idx for idx, cat in enumerate(categories)}
        self.idx_to_cat_name = {idx: cat["name"] for idx, cat in enumerate(categories)}
        self.num_classes = len(categories)
        print(f"[COCO] {self.num_classes} categories loaded")

        images_info = {img["id"]: img for img in coco_data["images"]}

        # Build image_id -> list of (category_id, area) annotations
        img_to_anns: Dict[int, List[Tuple[int, float]]] = {}
        for ann in coco_data["annotations"]:
            img_id = ann["image_id"]
            cat_id = ann["category_id"]
            area = ann.get("area", 0)
            if img_id not in img_to_anns:
                img_to_anns[img_id] = []
            img_to_anns[img_id].append((cat_id, area))

        self.samples: List[Tuple[str, int]] = []
        class_counts: Dict[int, int] = {i: 0 for i in range(self.num_classes)}

        for img_id, anns in img_to_anns.items():
            if img_id not in images_info:
                continue

            unique_cats = set(cat_id for cat_id, _ in anns)

            if self.single_label_only and len(unique_cats) > 1:
                continue  # Skip multi-label images

            # Compute total area per category
            cat_areas: Dict[int, float] = {}
            for cat_id, area in anns:
                cat_areas[cat_id] = cat_areas.get(cat_id, 0) + area

            # Primary category = largest area
            primary_cat_id = max(cat_areas.keys(), key=lambda c: cat_areas[c])
            class_idx = self.cat_id_to_idx[primary_cat_id]

            if self.max_samples_per_class is not None:
                if class_counts[class_idx] >= self.max_samples_per_class:
                    continue

            img_info = images_info[img_id]
            img_path = self.image_dir / img_info["file_name"]
            if img_path.exists():
                self.samples.append((str(img_path), class_idx))
                class_counts[class_idx] += 1

        # Store targets for compatibility with get_dataset_labels
        self.targets = [s[1] for s in self.samples]

        print(f"[COCO] Loaded {len(self.samples)} samples")
        non_empty = sum(1 for c in class_counts.values() if c > 0)
        print(f"[COCO] Classes with samples: {non_empty}/{self.num_classes}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def get_class_name(self, idx: int) -> str:
        """Get category name for a class index."""
        return self.idx_to_cat_name.get(idx, f"class_{idx}")


def build_transform(backbone, use_clip: bool) -> transforms.Compose:
    """
    Deterministic transform for feature extraction.
    - If CLIP backbone exposes `.preprocess`, use it.
    - Else, use ImageNet style preprocessing.
    """
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


def get_dataset(
    dataset_name: str,
    data_dir: str,
    transform: transforms.Compose,
    coco_single_label: bool = False,
    coco_max_per_class: Optional[int] = None,
) -> Tuple[Dataset, Dataset, int]:
    dataset_name = dataset_name.lower()

    if dataset_name == "cifar10":
        train_dataset = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform)
        test_dataset = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform)
        return train_dataset, test_dataset, 10

    if dataset_name == "cifar100":
        train_dataset = torchvision.datasets.CIFAR100(root=data_dir, train=True, download=True, transform=transform)
        test_dataset = torchvision.datasets.CIFAR100(root=data_dir, train=False, download=True, transform=transform)
        return train_dataset, test_dataset, 100

    if dataset_name == "tiny-imagenet":
        train_dataset = TinyImageNetDataset(root=data_dir, train=True, download=True, transform=transform)
        test_dataset = TinyImageNetDataset(root=data_dir, train=False, download=True, transform=transform)
        return train_dataset, test_dataset, 200

    if dataset_name in ("cub", "cub200", "cub-200"):
        # Expect prepared folder: data_dir/cub200/train and data_dir/cub200/test
        train_root = Path(data_dir) / "cub200" / "train"
        test_root = Path(data_dir) / "cub200" / "test"
        if not train_root.exists() or not test_root.exists():
            raise FileNotFoundError(
                f"CUB-200 folder not found.\nExpected:\n  {train_root}\n  {test_root}\n"
                f"Please run: python -m scripts.prepare_cub200 --data_root {data_dir}"
            )
        train_dataset = tv_datasets.ImageFolder(root=str(train_root), transform=transform)
        test_dataset = tv_datasets.ImageFolder(root=str(test_root), transform=transform)
        return train_dataset, test_dataset, 200

    if dataset_name == "coco":
        # COCO dataset for classification (80 classes)
        # Default path: data_dir/coco/ or data_dir itself if it contains train2017/
        coco_root = Path(data_dir)
        if not (coco_root / "train2017").exists():
            coco_root = coco_root / "coco"
        if not (coco_root / "train2017").exists():
            raise FileNotFoundError(
                f"COCO dataset not found.\nExpected structure:\n"
                f"  {coco_root}/train2017/\n"
                f"  {coco_root}/val2017/\n"
                f"  {coco_root}/annotations/instances_train2017.json\n"
                f"  {coco_root}/annotations/instances_val2017.json"
            )
        train_dataset = COCOClassificationDataset(
            root=str(coco_root),
            train=True,
            transform=transform,
            single_label_only=coco_single_label,
            max_samples_per_class=coco_max_per_class,
        )
        test_dataset = COCOClassificationDataset(
            root=str(coco_root),
            train=False,
            transform=transform,
            single_label_only=coco_single_label,
            max_samples_per_class=None,  # Don't limit test set
        )
        return train_dataset, test_dataset, 80

    raise ValueError(f"Unsupported dataset: {dataset_name}")


def get_dataset_labels(ds: Dataset) -> np.ndarray:
    """
    Get labels without triggering transforms.
    """
    if hasattr(ds, "targets"):
        return np.array(getattr(ds, "targets"))
    if hasattr(ds, "samples"):
        samples = getattr(ds, "samples")
        return np.array([s[1] for s in samples])
    # Read samples when the dataset does not expose a label array.
    labels = []
    for i in range(len(ds)):
        _, y = ds[i]
        labels.append(int(y))
    return np.array(labels)


# -----------------------------
# Feature extraction
# -----------------------------
@dataclass
class ExtractedFeatures:
    dense: Optional[torch.Tensor]          # [N, 2048]
    bottleneck: Optional[torch.Tensor]     # [N, D_bn]
    labels: torch.Tensor                   # [N]


@torch.no_grad()
def extract_features(
    dataloader: DataLoader,
    backbone: nn.Module,
    sae: nn.Module,
    device: torch.device,
    bottleneck_type: str,
    epsilon: float,
    need_dense: bool = True,
    need_bottleneck: bool = True,
    feature_dtype: str = "float32",
    gc_interval: int = 20,
) -> ExtractedFeatures:
    """
    Return dense and/or bottleneck features on CPU in the requested dtype.

    Transfer each batch to CPU and clear CUDA caches every gc_interval batches.
    """
    import gc
    
    backbone.eval()
    sae.eval()

    dense_list: List[torch.Tensor] = []
    bn_list: List[torch.Tensor] = []
    label_list: List[torch.Tensor] = []

    bottleneck_type = bottleneck_type.lower()
    assert bottleneck_type in ("global", "mc", "global_mc"), f"Unknown bottleneck_type: {bottleneck_type}"

    total_batches = len(dataloader)
    
    for batch_idx, (images, labels) in enumerate(tqdm(dataloader, desc="[Feat] Extracting", leave=False)):
        images = images.to(device, non_blocking=True)

        out = backbone(images)
        
        if need_dense:
            pooled = out["pooled_features"].detach()  # [B,2048]
            dense_list.append(to_dtype(pooled.cpu(), feature_dtype))
            del pooled

        if need_bottleneck:
            spatial = out["spatial_features"]  # [B,49,2048]
            # Match the SAE's float32 weights.
            if spatial.dtype != torch.float32:
                spatial = spatial.float()
            sae_out = sae(spatial)
            del spatial
            
            if bottleneck_type == "global":
                bn = sae_out["global_sparse"].detach()  # [B,C]
            elif bottleneck_type == "mc":
                bn = compute_channel_statistics(sae_out["sparse_activations"], epsilon=epsilon)  # [B,2C]
            else:  # global_mc
                global_sparse = sae_out["global_sparse"].detach()
                mc = compute_channel_statistics(sae_out["sparse_activations"], epsilon=epsilon)
                bn = torch.cat([global_sparse, mc], dim=1)  # [B, 3C]
                del global_sparse, mc

            bn_list.append(to_dtype(bn.cpu(), feature_dtype))
            del sae_out, bn

        label_list.append(labels.cpu())

        del images, out

        # Periodic memory cleanup
        if device.type == "cuda" and (batch_idx + 1) % gc_interval == 0:
            torch.cuda.empty_cache()
            gc.collect()

    if device.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()
    
    dense = torch.cat(dense_list, dim=0) if need_dense else None
    del dense_list
    
    bottleneck = torch.cat(bn_list, dim=0) if need_bottleneck else None
    del bn_list
    
    all_labels = torch.cat(label_list, dim=0).long()
    del label_list
    
    gc.collect()

    return ExtractedFeatures(dense=dense, bottleneck=bottleneck, labels=all_labels)


# -----------------------------
# Few-shot sampling
# -----------------------------
def sample_few_shot_indices(
    labels: np.ndarray,
    num_classes: int,
    shots: int,
    seed: int,
    allow_fewer: bool = True,
    verbose: bool = True,
) -> np.ndarray:
    """
    Balanced sampling: shots per class.
    
    Args:
        labels: Array of class labels for all samples
        num_classes: Total number of classes
        shots: Target number of samples per class
        seed: Random seed for reproducibility
        allow_fewer: If True, use all available samples when a class has fewer
                     than requested shots (instead of raising an error)
        verbose: If True, print warnings about classes with insufficient samples
    
    Returns:
        Array of sampled indices
    """
    rng = np.random.default_rng(seed)
    indices_per_class: List[np.ndarray] = []
    insufficient_classes: List[Tuple[int, int]] = []  # (class_id, available_count)
    empty_classes: List[int] = []
    
    for c in range(num_classes):
        cls_idx = np.where(labels == c)[0]
        
        if cls_idx.size == 0:
            empty_classes.append(c)
            continue  # Skip empty classes
        
        if shots > cls_idx.size:
            if allow_fewer:
                insufficient_classes.append((c, cls_idx.size))
                # Use all available samples for this class
                chosen = cls_idx.copy()
            else:
                raise ValueError(f"shots={shots} > available={cls_idx.size} for class {c}")
        else:
            chosen = rng.choice(cls_idx, size=shots, replace=False)
        
        indices_per_class.append(chosen)
    
    # Print warnings if verbose
    if verbose:
        if empty_classes:
            print(f"[Warning] {len(empty_classes)} classes have no samples, skipped: {empty_classes[:10]}{'...' if len(empty_classes) > 10 else ''}")
        if insufficient_classes:
            print(f"[Warning] {len(insufficient_classes)} classes have fewer than {shots} samples:")
            for cls_id, avail in insufficient_classes[:5]:
                print(f"  - Class {cls_id}: only {avail} samples (using all)")
            if len(insufficient_classes) > 5:
                print(f"  ... and {len(insufficient_classes) - 5} more classes")
    
    if not indices_per_class:
        raise ValueError("No valid samples found for any class!")
    
    return np.concatenate(indices_per_class)


# -----------------------------
# Linear and cosine heads
# -----------------------------
class LinearClassifier(nn.Module):
    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.fc = nn.Linear(input_dim, num_classes)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class CosineClassifier(nn.Module):
    """
    Cosine classifier: normalize x and weights, scale by temperature.
    """

    def __init__(self, input_dim: int, num_classes: int, temperature: float = 10.0, learnable_scale: bool = True):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(num_classes, input_dim))
        nn.init.xavier_uniform_(self.weight)
        if learnable_scale:
            self.logit_scale = nn.Parameter(torch.tensor(float(temperature)))
        else:
            self.register_buffer("logit_scale", torch.tensor(float(temperature)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.normalize(x, dim=1)
        w = F.normalize(self.weight, dim=1)
        scale = torch.clamp(self.logit_scale, min=1e-3)
        return scale * (x @ w.t())


# -----------------------------
# Head training (offline on extracted features)
# -----------------------------
def _l2_penalty(model: nn.Module) -> torch.Tensor:
    """
    Standard L2 penalty on weight-like parameters (exclude biases).
    """
    penalty = torch.tensor(0.0, device=next(model.parameters()).device)
    for name, p in model.named_parameters():
        if p.requires_grad and p.dim() >= 2:  # weights
            penalty = penalty + (p * p).sum()
    return penalty


def train_head_lbfgs(
    X: torch.Tensor,
    y: torch.Tensor,
    num_classes: int,
    head_type: str,
    l2: float,
    device: torch.device,
    max_iter: int = 200,
    lr: float = 1.0,
    cosine_temperature: float = 10.0,
    cosine_learnable_scale: bool = True,
    max_gpu_samples: int = 120000,
) -> nn.Module:
    """
    Train a linear or cosine head with full-batch LBFGS.

    Use CPU when a CUDA run exceeds max_gpu_samples, then return the head to
    the requested device for evaluation.
    """
    n_samples = X.shape[0]
    input_dim = X.shape[1]
    
    if n_samples > max_gpu_samples and device.type == "cuda":
        train_device = torch.device("cpu")
        print(f"[LBFGS] Dataset size {n_samples} > {max_gpu_samples}, training on CPU to save GPU memory")
    else:
        train_device = device
    
    X_train = X.to(device=train_device, dtype=torch.float32)
    y_train = y.to(device=train_device)

    head_type = head_type.lower()
    if head_type == "linear":
        model: nn.Module = LinearClassifier(input_dim, num_classes).to(train_device)
    elif head_type == "cosine":
        model = CosineClassifier(
            input_dim, num_classes, temperature=cosine_temperature, learnable_scale=cosine_learnable_scale
        ).to(train_device)
    else:
        raise ValueError(f"Unknown head_type: {head_type}")

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.LBFGS(
        model.parameters(), lr=lr, max_iter=max_iter, line_search_fn="strong_wolfe"
    )

    def closure():
        optimizer.zero_grad(set_to_none=True)
        logits = model(X_train)
        loss = criterion(logits, y_train)
        if l2 > 0:
            loss = loss + 0.5 * l2 * _l2_penalty(model)
        loss.backward()
        return loss

    optimizer.step(closure)
    
    # Move model back to original device for evaluation
    if train_device != device:
        model = model.to(device)
    
    return model


def train_head_sgd(
    X: torch.Tensor,
    y: torch.Tensor,
    num_classes: int,
    head_type: str,
    weight_decay: float,
    device: torch.device,
    epochs: int = 100,
    lr: float = 1e-2,
    batch_size: int = 256,
    cosine_temperature: float = 10.0,
    cosine_learnable_scale: bool = True,
) -> nn.Module:
    """
    Train a head with mini-batch Adam, transferring each batch from CPU to device.
    """
    X_cpu = X.cpu().float()
    y_cpu = y.cpu().long()

    input_dim = X_cpu.shape[1]
    head_type = head_type.lower()
    if head_type == "linear":
        model: nn.Module = LinearClassifier(input_dim, num_classes).to(device)
    elif head_type == "cosine":
        model = CosineClassifier(
            input_dim, num_classes, temperature=cosine_temperature, learnable_scale=cosine_learnable_scale
        ).to(device)
    else:
        raise ValueError(f"Unknown head_type: {head_type}")

    ds = TensorDataset(X_cpu, y_cpu)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=(device.type == "cuda"))

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

    return model


@torch.no_grad()
def evaluate_head(model: nn.Module, X: torch.Tensor, y: torch.Tensor, device: torch.device, batch_size: int = 1024) -> float:
    model.eval()
    ds = TensorDataset(X, y)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)

    correct = 0
    total = 0
    for xb, yb in loader:
        xb = xb.to(device=device, dtype=torch.float32)
        yb = yb.to(device=device)
        logits = model(xb)
        pred = logits.argmax(dim=1)
        correct += (pred == yb).sum().item()
        total += yb.numel()
    return 100.0 * correct / max(total, 1)


# -----------------------------
# Plotting
# -----------------------------
def plot_curves(
    shots_labels: List[str],
    curves: Dict[str, Dict[str, float]],
    output_path: Path,
    title: str,
):
    """
    curves: method -> {shot_label -> mean_acc}
    """
    import matplotlib.pyplot as plt

    x = np.arange(len(shots_labels))

    plt.figure(figsize=(8, 4.5))
    for method, by_shot in curves.items():
        y = [by_shot[s] for s in shots_labels]
        plt.plot(x, y, marker="o", label=method)

    plt.xticks(x, shots_labels)
    plt.xlabel("Shots per class")
    plt.ylabel("Accuracy (%)")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


# -----------------------------
# Main experiment
# -----------------------------
def parse_shots(s: str) -> List[Union[int, str]]:
    """
    Parse shots like "1,2,4,8,16,full" -> [1,2,4,8,16,'full']
    """
    parts = [p.strip() for p in s.split(",") if p.strip()]
    shots: List[Union[int, str]] = []
    for p in parts:
        if p.lower() == "full":
            shots.append("full")
        else:
            shots.append(int(p))
    return shots


def parse_int_list(s: str) -> List[int]:
    parts = [p.strip() for p in s.split(",") if p.strip()]
    return [int(p) for p in parts]


def parse_float_list(s: str) -> List[float]:
    parts = [p.strip() for p in s.split(",") if p.strip()]
    return [float(p) for p in parts]


def main():
    parser = argparse.ArgumentParser(description="Evaluate concept bottleneck completeness")

    # Dataset
    parser.add_argument("--dataset", type=str, default="cub200", 
                        choices=["cifar10", "cifar100", "tiny-imagenet", "cub200", "coco"],
                        help="Dataset to evaluate")
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Dataset root; CUB uses train/test, COCO uses train2017/val2017/annotations")
    
    # COCO-specific options
    parser.add_argument("--coco_single_label", type=str2bool, default=False,
                        help="Keep only COCO images annotated with a single category")
    parser.add_argument("--coco_max_per_class", type=int, default=0,
                        help="Maximum COCO training samples per class; 0 uses all")

    # Backbone
    parser.add_argument("--use_clip", type=str2bool, default=True, 
                        help="Use the CLIP visual backbone")
    parser.add_argument("--clip_variant", type=str, default="RN50", 
                        help="OpenAI CLIP model name")
    parser.add_argument("--clip_pretrained", type=str, default="openai", 
                        help="CLIP weight-source label for logs and cache tags")
    # SAE
    parser.add_argument(
        "--sae_checkpoint",
        type=str,
        default=None,
        help="Image SAE checkpoint path",
    )
    parser.add_argument(
        "--cub_sae_checkpoint",
        type=str,
        default=None,
        help=(
            "Optional checkpoint override for CUB-200; "
            "used instead of --sae_checkpoint when the file exists."
        ),
    )
    parser.add_argument("--sae_hidden_dim", type=int, default=8192,
                        help="Number of SAE latent features")
    parser.add_argument("--sae_k_sparse", type=int, default=64,
                        help="Rank used for the SAE activation threshold")

    # Bottleneck
    parser.add_argument("--bottleneck", type=str, default="mc", 
                        choices=["global", "mc", "global_mc"],
                        help="Feature type: global (summed activations), mc (max/count), or global_mc (concatenation)")
    parser.add_argument("--channel_epsilon", type=float, default=1e-8,
                        help="Activation threshold for spatial counts")

    # Normalization
    parser.add_argument("--dense_l2", type=str2bool, default=False, 
                        help="L2-normalize dense pooled features (LP baseline)")
    parser.add_argument("--bottleneck_l2", type=str2bool, default=False, 
                        help="L2-normalize bottleneck features after blocknorm")
    parser.add_argument("--blocknorm", type=str, default="none", 
                        choices=["none", "stat", "per_concept"],
                        help="Normalize max/count features by block statistics or per-concept statistics")
    parser.add_argument("--blocknorm_no_sqrt", action="store_true",
                        help="Skip the square-root transform on activation counts")
    parser.add_argument("--norm_fit_max_samples", type=int, default=0, 
                        help="Image limit for fitting normalization; 0 uses all training images")

    # Few-shot protocol
    parser.add_argument("--shots", type=str, default="1,2,4,8,16,full",
                        help="Comma-separated samples per class; 'full' uses the complete training set")
    parser.add_argument("--seeds", type=str, default="0,1,2",
                        help="Comma-separated seeds for repeated runs")

    # Feature extraction
    parser.add_argument("--batch_size_extract", type=int, default=128,
                        help="Batch size for feature extraction")
    parser.add_argument("--num_workers", type=int, default=0,
                        help="Number of data-loading workers")
    parser.add_argument("--feature_dtype", type=str, default="float32", 
                        choices=["float32", "float16", "bfloat16"],
                        help="Storage dtype for extracted features")
    parser.add_argument("--gc_interval", type=int, default=20,
                        help="CUDA cache and garbage-collection interval in extraction batches")

    # Head training
    parser.add_argument("--head_type", type=str, default="linear", 
                        choices=["linear", "cosine"],
                        help="Classification head: linear or scaled cosine similarity")
    parser.add_argument("--solver", type=str, default="lbfgs", 
                        choices=["lbfgs", "sgd"],
                        help="Training method: lbfgs (full batch) or sgd (mini-batch Adam)")
    parser.add_argument("--lbfgs_max_iter", type=int, default=1000,
                        help="Maximum LBFGS iterations")
    parser.add_argument("--lbfgs_lr", type=float, default=1.0,
                        help="LBFGS learning rate")
    parser.add_argument("--lbfgs_max_samples", type=int, default=120000,
                        help="Sample threshold for switching to mini-batch Adam; grid search uses CPU LBFGS above it")
    parser.add_argument("--l2", type=float, default=1e-2, 
                        help="L2 penalty for LBFGS or weight decay for mini-batch Adam")
    parser.add_argument(
        "--l2_lp",
        type=float,
        default=1e-5,
        help="L2 regularization for the linear probe",
    )
    parser.add_argument(
        "--l2_cbm",
        type=float,
        default=0.001,
        help="L2 regularization for the concept bottleneck head",
    )
    parser.add_argument(
        "--l2_search",
        type=str2bool,
        default=True,
        help="Run the LBFGS L2 grid search and print its results",
    )
    parser.add_argument(
        "--l2_search_grid",
        type=str,
        default="0,1e-6,1e-5,1e-4,1e-3,1e-2,1e-1,1",
        help="Comma-separated L2 values for LBFGS grid search",
    )
    parser.add_argument("--sgd_epochs", type=int, default=200,
                        help="Training epochs for mini-batch Adam")
    parser.add_argument("--sgd_lr", type=float, default=1e-2,
                        help="Learning rate for mini-batch Adam")
    parser.add_argument("--sgd_batch_size", type=int, default=128,
                        help="Batch size for mini-batch Adam")
    parser.add_argument("--cosine_temperature", type=float, default=10.0,
                        help="Initial scale for cosine classification logits")
    parser.add_argument("--cosine_learnable_scale", type=str2bool, default=True,
                        help="Learn the cosine classifier's logit scale")

    # Runtime
    parser.add_argument("--device", type=str, 
                        default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Compute device, such as cuda or cpu")
    parser.add_argument("--cache_dir", type=str, default="", 
                        help="Feature and normalizer cache directory; empty disables caching")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory for result JSON files and plots")
    parser.add_argument("--no_plot", action="store_true",
                        help="Skip curve plotting")

    args = parse_args_with_config(parser)

    missing_paths = [
        name
        for name in ("data_dir", "output_dir", "sae_checkpoint")
        if not str(getattr(args, name, "") or "").strip()
    ]
    if missing_paths:
        raise ValueError(
            "The following paths must be provided through --config or command-line "
            "arguments: " + ", ".join(missing_paths)
        )

    output_dir = Path(args.output_dir)
    if not args.l2_search:
        output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")

    # Resolve the dataset-specific SAE checkpoint.
    is_cub = args.dataset.lower() == "cub200"

    sae_checkpoint_path = args.sae_checkpoint
    if is_cub:
        if args.cub_sae_checkpoint and args.cub_sae_checkpoint.strip() != "":
            cub_sae_path = Path(args.cub_sae_checkpoint)
            if cub_sae_path.exists():
                sae_checkpoint_path = args.cub_sae_checkpoint
                print(f"[SAE] Using CUB checkpoint override: {sae_checkpoint_path}")
            else:
                print(f"[Warning] CUB checkpoint override not found: {cub_sae_path}")
                if args.sae_checkpoint and Path(args.sae_checkpoint).exists():
                    print(f"[Warning] Falling back to --sae_checkpoint: {args.sae_checkpoint}")
                    sae_checkpoint_path = args.sae_checkpoint
                else:
                    raise FileNotFoundError(
                        f"Neither SAE checkpoint was found.\n"
                        f"  --cub_sae_checkpoint: {args.cub_sae_checkpoint}\n"
                        f"  --sae_checkpoint: {args.sae_checkpoint}"
                    )
        else:
            if args.sae_checkpoint and Path(args.sae_checkpoint).exists():
                print(f"[SAE] Using --sae_checkpoint: {args.sae_checkpoint}")
            else:
                raise ValueError(
                    "CUB-200 evaluation requires an existing SAE checkpoint. Check --sae_checkpoint or --cub_sae_checkpoint."
                )
    else:
        if not sae_checkpoint_path or not Path(sae_checkpoint_path).exists():
            raise FileNotFoundError(
                f"SAE checkpoint not found: {sae_checkpoint_path}\n"
                f"Provide an existing file with --sae_checkpoint."
            )

    # Cache tags (avoid collisions between backbones and different SAE checkpoints)
    backbone_tag = f"clip{args.clip_variant}_{args.clip_pretrained}_base"
    sae_tag = f"sae{_stable_tag(sae_checkpoint_path)}"

    # Save args + resolved runtime config (skip in L2 search mode)
    if not args.l2_search:
        with open(output_dir / "config.json", "w", encoding="utf-8") as f:
            json.dump(vars(args), f, indent=2, ensure_ascii=False)
        with open(output_dir / "resolved.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "dataset": args.dataset,
                    "is_cub200": is_cub,
                    "sae_checkpoint_path": sae_checkpoint_path,
                    "backbone_tag": backbone_tag,
                    "sae_tag": sae_tag,
                },
                f,
                indent=2,
                ensure_ascii=False,
            )

    # 1) Build a temp backbone to get deterministic transform (esp. CLIP preprocess)
    print("[Init] Building backbone (temp) for transforms...")
    temp_backbone = create_resnet_backbone(
        args.use_clip,
        args.clip_variant,
        args.clip_pretrained,
    )
    transform = build_transform(temp_backbone, use_clip=args.use_clip)

    # 2) Load datasets
    print(f"[Data] Loading dataset: {args.dataset}")
    coco_max_per_class = args.coco_max_per_class if args.coco_max_per_class > 0 else None
    train_dataset, test_dataset, num_classes = get_dataset(
        args.dataset,
        args.data_dir,
        transform=transform,
        coco_single_label=args.coco_single_label,
        coco_max_per_class=coco_max_per_class,
    )

    train_labels_np = get_dataset_labels(train_dataset)

    # 3) Build actual backbone + SAE
    print("[Init] Building backbone + SAE...")
    backbone = create_resnet_backbone(
        args.use_clip,
        args.clip_variant,
        args.clip_pretrained,
    ).to(device)
    sae = create_image_sae(args.sae_hidden_dim, args.sae_k_sparse).to(device)
    load_image_sae_weights(sae_checkpoint_path, sae, map_location=device)

    # 4) DataLoaders (deterministic, no augmentation)
    train_loader_full = DataLoader(
        train_dataset,
        batch_size=args.batch_size_extract,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size_extract,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    # 5) Fit block normalizer if needed (only meaningful for 'mc' or for the mc-part of global_mc).
    normalizer: Optional[BlockNormalizer] = None
    if args.blocknorm != "none" and args.bottleneck in ("mc", "global_mc"):
        max_samples = args.norm_fit_max_samples if args.norm_fit_max_samples > 0 else None
        cache_dir = Path(args.cache_dir) if args.cache_dir else None
        norm_cache_path = None
        if cache_dir is not None:
            cache_dir.mkdir(parents=True, exist_ok=True)
            norm_cache_path = cache_dir / (
                f"{args.dataset}_{backbone_tag}_{sae_tag}_blocknorm_{args.blocknorm}"
                f"_eps{args.channel_epsilon}_hd{args.sae_hidden_dim}_sqrt{not args.blocknorm_no_sqrt}.pth"
            )

        if norm_cache_path is not None and norm_cache_path.exists():
            print(f"[Norm] Loading cached BlockNormalizer: {norm_cache_path}")
            state = torch.load(norm_cache_path, map_location="cpu", weights_only=False)
            normalizer = BlockNormalizer(
                mode=state["mode"], eps=state["eps"], apply_sqrt_on_counts=state["apply_sqrt_on_counts"]
            )
            normalizer.mu = state["mu"]
            normalizer.sigma = state["sigma"]
            normalizer.fitted = True
        else:
            print(f"[Norm] Fitting BlockNormalizer: mode={args.blocknorm}, sqrt_on_counts={not args.blocknorm_no_sqrt}")
            normalizer = BlockNormalizer(
                mode=args.blocknorm, eps=1e-8, apply_sqrt_on_counts=(not args.blocknorm_no_sqrt)
            )
            normalizer.fit_from_loader(
                train_loader=train_loader_full,
                backbone=backbone,
                sae=sae,
                epsilon=args.channel_epsilon,
                device=device,
                max_samples=max_samples,
            )
            if norm_cache_path is not None:
                torch.save(
                    {
                        "mode": normalizer.mode,
                        "eps": normalizer.eps,
                        "apply_sqrt_on_counts": normalizer.apply_sqrt_on_counts,
                        "mu": normalizer.mu,
                        "sigma": normalizer.sigma,
                    },
                    norm_cache_path,
                )
                print(f"[Norm] Saved BlockNormalizer cache: {norm_cache_path}")

    # 6) Extract and (optionally) cache test features for both lines (LP + MetaphorCBM-PA)
    cache_dir = Path(args.cache_dir) if args.cache_dir else None
    test_cache_path = None
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        test_cache_path = cache_dir / (
            f"{args.dataset}_{backbone_tag}_{sae_tag}_test_dense_bn_{args.bottleneck}"
            f"_hd{args.sae_hidden_dim}_eps{args.channel_epsilon}_dtype{args.feature_dtype}.pth"
        )

    if test_cache_path is not None and test_cache_path.exists():
        print(f"[Cache] Loading test features: {test_cache_path}")
        test_pack = torch.load(test_cache_path, map_location="cpu", weights_only=False)
        X_test_dense = test_pack["dense"]
        X_test_bn = test_pack["bottleneck"]
        y_test = test_pack["labels"]
    else:
        print("[Feat] Extracting test features (once)...")
        test_feats = extract_features(
            dataloader=test_loader,
            backbone=backbone,
            sae=sae,
            device=device,
            bottleneck_type=args.bottleneck,
            epsilon=args.channel_epsilon,
            need_dense=True,
            need_bottleneck=True,
            feature_dtype=args.feature_dtype,
            gc_interval=args.gc_interval,
        )
        X_test_dense = test_feats.dense
        X_test_bn = test_feats.bottleneck
        y_test = test_feats.labels

        if test_cache_path is not None:
            torch.save(
                {
                    "dense": X_test_dense,
                    "bottleneck": X_test_bn,
                    "labels": y_test,
                    "meta": {
                        "args": vars(args),
                        "backbone_tag": backbone_tag,
                        "sae_tag": sae_tag,
                        "sae_checkpoint_path": sae_checkpoint_path,
                    },
                },
                test_cache_path,
            )
            print(f"[Cache] Saved test features: {test_cache_path}")

    assert X_test_dense is not None and X_test_bn is not None

    # Apply normalizations to test features
    if args.dense_l2:
        X_test_dense = to_dtype(l2_normalize(X_test_dense.float(), dim=1).cpu(), args.feature_dtype)

    if normalizer is not None:
        if args.bottleneck == "mc":
            X_test_bn = normalizer.transform(X_test_bn).cpu()
            if args.bottleneck_l2:
                X_test_bn = l2_normalize(X_test_bn, dim=1).cpu()
            X_test_bn = to_dtype(X_test_bn, args.feature_dtype)
        else:  # global_mc: normalize only the mc part
            C = args.sae_hidden_dim
            Xg = X_test_bn[:, :C].float()
            Xmc = X_test_bn[:, C:].float()
            Xmc = normalizer.transform(Xmc).cpu()
            if args.bottleneck_l2:
                Xmc = l2_normalize(Xmc, dim=1).cpu()
            X_test_bn = torch.cat([Xg.cpu(), Xmc.cpu()], dim=1)
            X_test_bn = to_dtype(X_test_bn, args.feature_dtype)

    # 7) L2 grid search mode (LBFGS only)
    if args.l2_search:
        if args.solver != "lbfgs":
            print("[L2Search] Using LBFGS regardless of --solver")
        l2_grid = parse_float_list(args.l2_search_grid)
        print(f"[L2Search] L2 grid: {l2_grid}")
        if len(l2_grid) == 0:
            raise ValueError("L2 search grid is empty.")

        shots_list = parse_shots(args.shots)
        seeds = parse_int_list(args.seeds)

        # results[method][shot_label][l2] = list(accs over seeds)
        results: Dict[str, Dict[str, Dict[float, List[float]]]] = {
            "LP": {},
            "MetaphorCBM-PA": {},
        }

        for shot in shots_list:
            shot_label = "full" if shot == "full" else str(int(shot))
            results["LP"][shot_label] = {l2: [] for l2 in l2_grid}
            results["MetaphorCBM-PA"][shot_label] = {l2: [] for l2 in l2_grid}

            for seed in seeds:
                set_seed(seed)
                print(f"\n[L2Search] shot={shot_label}, seed={seed}")

                # sample indices
                if shot == "full":
                    indices = np.arange(len(train_dataset))
                else:
                    indices = sample_few_shot_indices(
                        train_labels_np, num_classes=num_classes, shots=int(shot), seed=seed
                    )

                subset = Subset(train_dataset, indices.tolist())
                subset_loader = DataLoader(
                    subset,
                    batch_size=min(args.batch_size_extract, len(subset)),
                    shuffle=False,
                    num_workers=args.num_workers,
                    pin_memory=(device.type == "cuda"),
                )

                # extract subset features (dense + bn) in one pass
                train_feats = extract_features(
                    dataloader=subset_loader,
                    backbone=backbone,
                    sae=sae,
                    device=device,
                    bottleneck_type=args.bottleneck,
                    epsilon=args.channel_epsilon,
                    need_dense=True,
                    need_bottleneck=True,
                    feature_dtype=args.feature_dtype,
                    gc_interval=args.gc_interval,
                )
                X_train_dense = train_feats.dense
                X_train_bn = train_feats.bottleneck
                y_train = train_feats.labels

                assert X_train_dense is not None and X_train_bn is not None

                # normalize train features the same way as test
                if args.dense_l2:
                    X_train_dense = to_dtype(l2_normalize(X_train_dense.float(), dim=1).cpu(), args.feature_dtype)

                if normalizer is not None:
                    if args.bottleneck == "mc":
                        X_train_bn = normalizer.transform(X_train_bn).cpu()
                        if args.bottleneck_l2:
                            X_train_bn = l2_normalize(X_train_bn, dim=1).cpu()
                        X_train_bn = to_dtype(X_train_bn, args.feature_dtype)
                    else:
                        C = args.sae_hidden_dim
                        Xg = X_train_bn[:, :C].float()
                        Xmc = X_train_bn[:, C:].float()
                        Xmc = normalizer.transform(Xmc).cpu()
                        if args.bottleneck_l2:
                            Xmc = l2_normalize(Xmc, dim=1).cpu()
                        X_train_bn = torch.cat([Xg.cpu(), Xmc.cpu()], dim=1)
                        X_train_bn = to_dtype(X_train_bn, args.feature_dtype)

                n_train_samples = X_train_dense.shape[0]
                if n_train_samples > args.lbfgs_max_samples:
                    print(
                        f"[L2Search] {n_train_samples} samples exceed lbfgs_max_samples={args.lbfgs_max_samples}."
                        " Continuing with LBFGS; CUDA runs use CPU training above this threshold."
                    )

                for l2 in l2_grid:
                    # LP
                    lp_head = train_head_lbfgs(
                        X_train_dense,
                        y_train,
                        num_classes=num_classes,
                        head_type=args.head_type,
                        l2=l2,
                        device=device,
                        max_iter=args.lbfgs_max_iter,
                        lr=args.lbfgs_lr,
                        cosine_temperature=args.cosine_temperature,
                        cosine_learnable_scale=args.cosine_learnable_scale,
                        max_gpu_samples=args.lbfgs_max_samples,
                    )
                    lp_acc = evaluate_head(lp_head, X_test_dense, y_test, device=device, batch_size=1024)
                    results["LP"][shot_label][l2].append(lp_acc)

                    # CBM
                    sae_head = train_head_lbfgs(
                        X_train_bn,
                        y_train,
                        num_classes=num_classes,
                        head_type=args.head_type,
                        l2=l2,
                        device=device,
                        max_iter=args.lbfgs_max_iter,
                        lr=args.lbfgs_lr,
                        cosine_temperature=args.cosine_temperature,
                        cosine_learnable_scale=args.cosine_learnable_scale,
                        max_gpu_samples=args.lbfgs_max_samples,
                    )
                    sae_acc = evaluate_head(sae_head, X_test_bn, y_test, device=device, batch_size=1024)
                    results["MetaphorCBM-PA"][shot_label][l2].append(sae_acc)

                    # Cleanup per L2
                    del lp_head, sae_head
                    if device.type == "cuda":
                        torch.cuda.empty_cache()

                # Cleanup per seed
                del X_train_dense, X_train_bn, y_train, train_feats
                if device.type == "cuda":
                    torch.cuda.empty_cache()

        # Print summary
        print("\n=== L2 Grid Search Summary (mean ± std over seeds) ===")
        for method, by_shot in results.items():
            print(f"\n[{method}]")
            for shot in shots_list:
                shot_label = "full" if shot == "full" else str(int(shot))
                best_l2 = None
                best_mean = -1.0
                best_std = 0.0
                print(f"  shots={shot_label}")
                for l2 in l2_grid:
                    accs = by_shot[shot_label][l2]
                    arr = np.array(accs, dtype=np.float64)
                    mean = float(arr.mean()) if arr.size > 0 else 0.0
                    std = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
                    print(f"    l2={l2:g}: {mean:6.2f} ± {std:5.2f} (n={arr.size})")
                    if mean > best_mean:
                        best_mean = mean
                        best_std = std
                        best_l2 = l2
                if best_l2 is not None:
                    print(f"    -> best l2={best_l2:g}: {best_mean:6.2f} ± {best_std:5.2f}")

        print("\n[L2Search] Done.")
        return

    # 7) Few-shot loop
    shots_list = parse_shots(args.shots)
    seeds = parse_int_list(args.seeds)
    l2_lp = args.l2 if args.l2_lp is None else args.l2_lp
    l2_cbm = args.l2 if args.l2_cbm is None else args.l2_cbm
    full_train_cache: Optional[Dict[str, torch.Tensor]] = None

    # results[method][shot_label] = list(acc over seeds)
    results: Dict[str, Dict[str, List[float]]] = {
        "LP": {},
        "MetaphorCBM-PA": {},
    }

    for shot in shots_list:
        shot_label = "full" if shot == "full" else str(int(shot))
        results["LP"][shot_label] = []
        results["MetaphorCBM-PA"][shot_label] = []

        for seed in seeds:
            set_seed(seed)
            print(f"\n[Run] shot={shot_label}, seed={seed}")

            used_full_cache = False
            if shot == "full" and full_train_cache is not None:
                X_train_dense = full_train_cache["dense"]
                X_train_bn = full_train_cache["bottleneck"]
                y_train = full_train_cache["labels"]
                used_full_cache = True
            else:
                # sample indices
                if shot == "full":
                    indices = np.arange(len(train_dataset))
                else:
                    indices = sample_few_shot_indices(
                        train_labels_np, num_classes=num_classes, shots=int(shot), seed=seed
                    )

                subset = Subset(train_dataset, indices.tolist())
                subset_loader = DataLoader(
                    subset,
                    batch_size=min(args.batch_size_extract, len(subset)),
                    shuffle=False,
                    num_workers=args.num_workers,
                    pin_memory=(device.type == "cuda"),
                )

                # extract subset features (dense + bn) in one pass
                train_feats = extract_features(
                    dataloader=subset_loader,
                    backbone=backbone,
                    sae=sae,
                    device=device,
                    bottleneck_type=args.bottleneck,
                    epsilon=args.channel_epsilon,
                    need_dense=True,
                    need_bottleneck=True,
                    feature_dtype=args.feature_dtype,
                    gc_interval=args.gc_interval,
                )
                X_train_dense = train_feats.dense
                X_train_bn = train_feats.bottleneck
                y_train = train_feats.labels

                assert X_train_dense is not None and X_train_bn is not None

                # normalize train features the same way as test
                if args.dense_l2:
                    X_train_dense = to_dtype(l2_normalize(X_train_dense.float(), dim=1).cpu(), args.feature_dtype)

                if normalizer is not None:
                    if args.bottleneck == "mc":
                        X_train_bn = normalizer.transform(X_train_bn).cpu()
                        if args.bottleneck_l2:
                            X_train_bn = l2_normalize(X_train_bn, dim=1).cpu()
                        X_train_bn = to_dtype(X_train_bn, args.feature_dtype)
                    else:
                        C = args.sae_hidden_dim
                        Xg = X_train_bn[:, :C].float()
                        Xmc = X_train_bn[:, C:].float()
                        Xmc = normalizer.transform(Xmc).cpu()
                        if args.bottleneck_l2:
                            Xmc = l2_normalize(Xmc, dim=1).cpu()
                        X_train_bn = torch.cat([Xg.cpu(), Xmc.cpu()], dim=1)
                        X_train_bn = to_dtype(X_train_bn, args.feature_dtype)

                if shot == "full":
                    full_train_cache = {
                        "dense": X_train_dense,
                        "bottleneck": X_train_bn,
                        "labels": y_train,
                    }

            # Use mini-batch Adam above the full-batch sample threshold.
            n_train_samples = X_train_dense.shape[0]
            use_lbfgs = (args.solver == "lbfgs") and (n_train_samples <= args.lbfgs_max_samples)
            if args.solver == "lbfgs" and n_train_samples > args.lbfgs_max_samples:
                print(f"[Solver] {n_train_samples} samples exceed lbfgs_max_samples={args.lbfgs_max_samples}; switching to mini-batch Adam")

            # Train & eval: LP
            if use_lbfgs:
                lp_head = train_head_lbfgs(
                    X_train_dense,
                    y_train,
                    num_classes=num_classes,
                    head_type=args.head_type,
                    l2=l2_lp,
                    device=device,
                    max_iter=args.lbfgs_max_iter,
                    lr=args.lbfgs_lr,
                    cosine_temperature=args.cosine_temperature,
                    cosine_learnable_scale=args.cosine_learnable_scale,
                )
            else:
                lp_head = train_head_sgd(
                    X_train_dense,
                    y_train,
                    num_classes=num_classes,
                    head_type=args.head_type,
                    weight_decay=l2_lp,
                    device=device,
                    epochs=args.sgd_epochs,
                    lr=args.sgd_lr,
                    batch_size=args.sgd_batch_size,
                    cosine_temperature=args.cosine_temperature,
                    cosine_learnable_scale=args.cosine_learnable_scale,
                )

            lp_acc = evaluate_head(lp_head, X_test_dense, y_test, device=device, batch_size=1024)
            results["LP"][shot_label].append(lp_acc)
            print(f"[LP] accuracy = {lp_acc:.2f}%")

            # Train & eval: MetaphorCBM-PA
            if use_lbfgs:
                sae_head = train_head_lbfgs(
                    X_train_bn,
                    y_train,
                    num_classes=num_classes,
                    head_type=args.head_type,
                    l2=l2_cbm,
                    device=device,
                    max_iter=args.lbfgs_max_iter,
                    lr=args.lbfgs_lr,
                    cosine_temperature=args.cosine_temperature,
                    cosine_learnable_scale=args.cosine_learnable_scale,
                )
            else:
                sae_head = train_head_sgd(
                    X_train_bn,
                    y_train,
                    num_classes=num_classes,
                    head_type=args.head_type,
                    weight_decay=l2_cbm,
                    device=device,
                    epochs=args.sgd_epochs,
                    lr=args.sgd_lr,
                    batch_size=args.sgd_batch_size,
                    cosine_temperature=args.cosine_temperature,
                    cosine_learnable_scale=args.cosine_learnable_scale,
                )

            sae_acc = evaluate_head(sae_head, X_test_bn, y_test, device=device, batch_size=1024)
            results["MetaphorCBM-PA"][shot_label].append(sae_acc)
            print(f"[MetaphorCBM-PA] accuracy = {sae_acc:.2f}%")
            
            # Memory cleanup after each seed iteration
            del lp_head, sae_head
            if not used_full_cache:
                del X_train_dense, X_train_bn, y_train, train_feats
            if device.type == "cuda":
                torch.cuda.empty_cache()

    # 8) Summaries
    summary: Dict[str, Dict[str, Dict[str, float]]] = {}
    for method, by_shot in results.items():
        summary[method] = {}
        for shot_label, accs in by_shot.items():
            arr = np.array(accs, dtype=np.float64)
            summary[method][shot_label] = {
                "mean": float(arr.mean()),
                "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
                "n": int(arr.size),
            }

    # Save raw + summary
    with open(output_dir / "results_raw.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    with open(output_dir / "results_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # Print table-like summary
    print("\n=== Summary (mean ± std over seeds) ===")
    for method in summary:
        print(f"\n[{method}]")
        for shot in parse_shots(args.shots):
            shot_label = "full" if shot == "full" else str(int(shot))
            m = summary[method][shot_label]["mean"]
            s = summary[method][shot_label]["std"]
            print(f"  shots={shot_label:>4s}: {m:6.2f} ± {s:5.2f}")

    # Plot
    if not args.no_plot:
        shots_labels = ["full" if s == "full" else str(int(s)) for s in shots_list]
        curves_mean = {
            "LP": {k: summary["LP"][k]["mean"] for k in shots_labels},
            "MetaphorCBM-PA": {k: summary["MetaphorCBM-PA"][k]["mean"] for k in shots_labels},
        }
        fig_path = output_dir / "fewshot_curve.png"
        plot_curves(shots_labels, curves_mean, fig_path, title=f"{args.dataset.upper()} Few-shot (LP vs MetaphorCBM-PA)")
        print(f"\n[Plot] Saved: {fig_path}")

    print(f"\n[Done] Outputs in: {output_dir.resolve()}")
