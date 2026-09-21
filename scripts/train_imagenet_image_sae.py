#!/usr/bin/env python3
"""Train an ImageNet image SAE on frozen CLIP-RN50 spatial features.

The training path is:

    frozen CLIP-RN50 layer4 spatial features -> ImageSAE -> reconstruction

The saved checkpoint keeps the same nested shape expected by downstream
loaders:

    checkpoint["model_state_dict"]["image_sae"]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import ImageFile
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder
from tqdm import tqdm

from metaphorcbm.config import parse_args_with_config

ImageFile.LOAD_TRUNCATED_IMAGES = True

from metaphorcbm.models import ImageSAE as FrameworkImageSAE
from metaphorcbm.models import ResNetWithHooks, create_clip_resnet_backbone


def str2bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    if v is None:
        return False
    s = str(v).strip().lower()
    if s in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if s in {"0", "false", "f", "no", "n", "off"}:
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
        json.dump(payload, f, indent=2, ensure_ascii=False)
    tmp.replace(path)


def append_jsonl(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def torch_load(path: Path, map_location: str | torch.device = "cpu") -> Any:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def sha256_file(path: Path, max_bytes: Optional[int] = None) -> str:
    h = hashlib.sha256()
    read_bytes = 0
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            if max_bytes is not None and read_bytes + len(chunk) > max_bytes:
                chunk = chunk[: max_bytes - read_bytes]
            h.update(chunk)
            read_bytes += len(chunk)
            if max_bytes is not None and read_bytes >= max_bytes:
                break
    return h.hexdigest()


def make_loader(
    dataset: ImageFolder | Subset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
    drop_last: bool,
) -> DataLoader:
    kwargs: Dict[str, Any] = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "drop_last": drop_last,
        "persistent_workers": num_workers > 0,
    }
    if num_workers > 0:
        kwargs["prefetch_factor"] = 4
    return DataLoader(dataset, **kwargs)


def maybe_limit_dataset(dataset: ImageFolder, max_samples: Optional[int], seed: int) -> ImageFolder | Subset:
    if max_samples is None or max_samples <= 0 or max_samples >= len(dataset):
        return dataset
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(dataset), size=max_samples, replace=False)
    return Subset(dataset, indices.tolist())


def load_imagenet_datasets(
    data_root: Path,
    train_transform: Any,
    val_transform: Any,
    train_max_samples: Optional[int],
    val_max_samples: Optional[int],
    seed: int,
) -> Tuple[ImageFolder | Subset, ImageFolder | Subset, ImageFolder, ImageFolder]:
    train_dir = data_root / "train"
    val_dir = data_root / "val"
    if not train_dir.is_dir():
        raise FileNotFoundError(f"Missing ImageNet train directory: {train_dir}")
    if not val_dir.is_dir():
        raise FileNotFoundError(f"Missing ImageNet val directory: {val_dir}")

    full_train = ImageFolder(train_dir, transform=train_transform)
    full_val = ImageFolder(val_dir, transform=val_transform)
    if full_train.class_to_idx != full_val.class_to_idx:
        raise RuntimeError("ImageFolder class_to_idx differs between train and val.")

    train = maybe_limit_dataset(full_train, train_max_samples, seed)
    val = maybe_limit_dataset(full_val, val_max_samples, seed + 1)
    return train, val, full_train, full_val


def build_backbone(args: argparse.Namespace, device: torch.device) -> nn.Module:
    if args.use_clip:
        print(f"[Backbone] CLIP-{args.clip_variant} ({args.clip_pretrained})")
        backbone = create_clip_resnet_backbone(
            {
                "clip_variant": args.clip_variant,
                "pretrained": args.clip_pretrained,
                "hook_layer": args.hook_layer,
                "freeze_backbone": True,
                "return_features": False,
                "finetuned_model_path": args.finetuned_model_path,
            }
        )
    else:
        print(f"[Backbone] torchvision ResNet-50 ({args.resnet_weights})")
        backbone = ResNetWithHooks(
            weights=args.resnet_weights,
            hook_layer=args.hook_layer,
            freeze_backbone=True,
            return_features=False,
        )
    backbone.to(device).eval()
    for p in backbone.parameters():
        p.requires_grad = False
    return backbone


def get_image_transform(backbone: nn.Module, args: argparse.Namespace, train: bool) -> Any:
    if args.use_clip and hasattr(backbone, "preprocess") and backbone.preprocess is not None:
        return backbone.preprocess

    from torchvision import transforms

    return transforms.Compose(
        [
            transforms.Resize(args.resize_size),
            transforms.CenterCrop(args.crop_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


def build_image_sae(args: argparse.Namespace, device: torch.device) -> nn.Module:
    sae = FrameworkImageSAE(
        input_dim=args.img_input_dim,
        hidden_dim=args.img_hidden_dim,
        k_sparse=args.img_k_sparse,
        use_bias=args.use_bias,
        activation=args.activation,
        initialization=args.initialization,
    )
    sae.to(device)
    return sae


def image_ue_loss(global_sparse: torch.Tensor, even_weight: float, eps: float = 1e-8) -> torch.Tensor:
    if global_sparse.dim() != 2:
        global_sparse = global_sparse.view(global_sparse.size(0), -1)
    batch_size, num_dims = global_sparse.shape
    if batch_size <= 1 or num_dims <= 1:
        return global_sparse.new_tensor(0.0)

    centered = global_sparse - global_sparse.mean(dim=0, keepdim=True)
    var = centered.pow(2).mean(dim=0) + eps
    std = var.sqrt()
    normalized = centered / std.clamp_min(eps)
    gram = normalized @ normalized.t()
    diag_counts = normalized.pow(2).sum(dim=0)
    offdiag_sq = gram.pow(2).sum() - diag_counts.pow(2).sum()
    denom = (max(batch_size - 1, 1) ** 2) * (num_dims * max(num_dims - 1, 1) + eps)
    uncorr = offdiag_sq / denom

    var_norm = var / (var.mean() + eps)
    even = (var_norm - 1.0).pow(2).mean()
    return uncorr + even_weight * even


def amp_context(device: torch.device, enabled: bool, dtype_name: str):
    if not enabled or device.type != "cuda":
        return torch.autocast(device_type=device.type, enabled=False)
    dtype = torch.float16 if dtype_name == "float16" else torch.bfloat16
    return torch.autocast(device_type="cuda", dtype=dtype, enabled=True)


def make_grad_scaler(device: torch.device, enabled: bool):
    use_scaler = bool(enabled and device.type == "cuda")
    try:
        return torch.amp.GradScaler("cuda", enabled=use_scaler)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=use_scaler)


class AverageMeter:
    def __init__(self) -> None:
        self.total = 0.0
        self.count = 0

    def update(self, value: float, n: int) -> None:
        self.total += float(value) * int(n)
        self.count += int(n)

    @property
    def avg(self) -> float:
        if self.count == 0:
            return 0.0
        return self.total / self.count


def batch_stats(outputs: Dict[str, torch.Tensor]) -> Dict[str, float]:
    sparse = outputs["sparse_activations"]
    global_sparse = outputs["global_sparse"]
    return {
        "sparse_nonzero_ratio": (sparse > 0).float().mean().item(),
        "sparse_mean_activation": sparse.abs().mean().item(),
        "patch_active_dims": (sparse > 0).sum(dim=-1).float().mean().item(),
        "global_active_dims": (global_sparse > 0).sum(dim=-1).float().mean().item(),
    }


def run_epoch(
    *,
    phase: str,
    epoch: int,
    loader: DataLoader,
    backbone: nn.Module,
    image_sae: nn.Module,
    optimizer: Optional[optim.Optimizer],
    scaler: Any,
    args: argparse.Namespace,
    device: torch.device,
) -> Dict[str, float]:
    is_train = phase == "train"
    image_sae.train(is_train)
    backbone.eval()

    meters = {
        "total_loss": AverageMeter(),
        "img_rec_loss": AverageMeter(),
        "ue_loss": AverageMeter(),
        "sparse_nonzero_ratio": AverageMeter(),
        "sparse_mean_activation": AverageMeter(),
        "patch_active_dims": AverageMeter(),
        "global_active_dims": AverageMeter(),
    }

    if is_train and optimizer is not None:
        optimizer.zero_grad(set_to_none=True)

    progress = tqdm(loader, desc=f"{phase} epoch {epoch}", dynamic_ncols=True)
    start_time = time.time()

    for step, (images, _) in enumerate(progress):
        images = images.to(device, non_blocking=args.pin_memory)
        batch_size = images.size(0)

        with torch.no_grad():
            with amp_context(device, args.amp, args.amp_dtype):
                backbone_outputs = backbone(images)
                spatial_features = backbone_outputs["spatial_features"].detach()

        with torch.set_grad_enabled(is_train):
            with amp_context(device, args.amp, args.amp_dtype):
                outputs = image_sae(spatial_features)
                rec_loss = F.mse_loss(outputs["reconstructed"], outputs["input"])
                if args.lambda_ue > 0:
                    ue = image_ue_loss(outputs["global_sparse"], args.ue_even_weight)
                else:
                    ue = rec_loss.new_tensor(0.0)
                loss = rec_loss + args.lambda_ue * ue

        if is_train and optimizer is not None:
            loss_to_backward = loss / args.accumulation_steps
            if args.amp:
                scaler.scale(loss_to_backward).backward()
            else:
                loss_to_backward.backward()

            should_step = (step + 1) % args.accumulation_steps == 0 or (step + 1) == len(loader)
            if should_step:
                if args.amp:
                    scaler.unscale_(optimizer)
                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(image_sae.parameters(), args.grad_clip)
                if args.amp:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)

        meters["total_loss"].update(loss.item(), batch_size)
        meters["img_rec_loss"].update(rec_loss.item(), batch_size)
        meters["ue_loss"].update(ue.item(), batch_size)
        stat_values = batch_stats(outputs)
        for key, value in stat_values.items():
            meters[key].update(value, batch_size)

        progress.set_postfix(
            {
                "loss": f"{meters['total_loss'].avg:.4f}",
                "rec": f"{meters['img_rec_loss'].avg:.4f}",
                "act": f"{meters['patch_active_dims'].avg:.1f}",
            }
        )

    elapsed = time.time() - start_time
    metrics = {key: meter.avg for key, meter in meters.items()}
    metrics["elapsed_sec"] = elapsed
    return metrics


def extract_image_sae_state(checkpoint: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    if "model_state_dict" in checkpoint and isinstance(checkpoint["model_state_dict"], dict):
        model_state = checkpoint["model_state_dict"]
        if "image_sae" in model_state and isinstance(model_state["image_sae"], dict):
            return model_state["image_sae"]
        stripped = {
            key[len("image_sae.") :]: value
            for key, value in model_state.items()
            if str(key).startswith("image_sae.")
        }
        if stripped:
            return stripped
    if "image_sae_state_dict" in checkpoint and isinstance(checkpoint["image_sae_state_dict"], dict):
        return checkpoint["image_sae_state_dict"]
    return checkpoint


def save_checkpoint(
    path: Path,
    *,
    epoch: int,
    image_sae: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: Optional[CosineAnnealingLR],
    scaler: Any,
    metrics: Dict[str, float],
    args: argparse.Namespace,
    class_to_idx: Dict[str, int],
    best_score: float,
    best_epoch: int,
    epochs_since_improvement: int,
    is_best: bool,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, Any] = {
        "epoch": int(epoch),
        "model_state_dict": {"image_sae": image_sae.state_dict()},
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
        "metrics": metrics,
        "config": vars(args),
        "class_to_idx": class_to_idx,
        "best_score": float(best_score),
        "best_epoch": int(best_epoch),
        "epochs_since_improvement": int(epochs_since_improvement),
        "is_best": bool(is_best),
        "timestamp": time.time(),
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def load_resume_state(
    resume_path: Path,
    image_sae: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: Optional[CosineAnnealingLR],
    scaler: Any,
    device: torch.device,
    model_only: bool,
) -> Tuple[int, float, int, int]:
    print(f"[Resume] Loading {resume_path}")
    checkpoint = torch_load(resume_path, map_location=device)
    image_sae.load_state_dict(extract_image_sae_state(checkpoint), strict=True)

    start_epoch = int(checkpoint.get("epoch", -1)) + 1 if isinstance(checkpoint, dict) else 0
    best_score = float(checkpoint.get("best_score", float("inf"))) if isinstance(checkpoint, dict) else float("inf")
    best_epoch = int(checkpoint.get("best_epoch", -1)) if isinstance(checkpoint, dict) else -1
    stale_epochs = int(checkpoint.get("epochs_since_improvement", 0)) if isinstance(checkpoint, dict) else 0

    if not model_only and isinstance(checkpoint, dict):
        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if scheduler is not None and checkpoint.get("scheduler_state_dict") is not None:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if scaler is not None and checkpoint.get("scaler_state_dict") is not None:
            scaler.load_state_dict(checkpoint["scaler_state_dict"])

    print(f"[Resume] start_epoch={start_epoch}, best_epoch={best_epoch}, best_score={best_score:.6f}")
    return start_epoch, best_score, best_epoch, stale_epochs


def prepare_output_dir(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    ckpt_dir = output_dir / "checkpoints"
    if (
        ckpt_dir.exists()
        and any(ckpt_dir.glob("*.pth"))
        and not args.resume
        and not args.overwrite
    ):
        raise FileExistsError(
            f"Checkpoint files already exist in {ckpt_dir}. "
            "Pass --resume or use --overwrite true if this is intentional."
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "logs").mkdir(exist_ok=True)
    (output_dir / "checkpoints").mkdir(exist_ok=True)
    (output_dir / "source").mkdir(exist_ok=True)

    source_path = Path(__file__).resolve()
    shutil.copy2(source_path, output_dir / "source" / source_path.name)

    config_payload = {
        "args": vars(args),
        "script_path": str(source_path),
        "script_sha256": sha256_file(source_path),
        "created_at": time.time(),
    }
    write_json(output_dir / "config.json", config_payload)


def train(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    args.output_dir = str(Path(args.output_dir))
    prepare_output_dir(args)

    print(f"[Device] {device}")
    print(f"[Data] {args.data_root}")
    print(f"[Output] {args.output_dir}")

    backbone = build_backbone(args, device)
    image_sae = build_image_sae(args, device)

    train_transform = get_image_transform(backbone, args, train=True)
    val_transform = get_image_transform(backbone, args, train=False)
    train_dataset, val_dataset, full_train, full_val = load_imagenet_datasets(
        Path(args.data_root),
        train_transform,
        val_transform,
        args.train_max_samples,
        args.val_max_samples,
        args.seed,
    )
    print(f"[Dataset] train={len(train_dataset)} val={len(val_dataset)} classes={len(full_train.classes)}")

    class_payload = {
        "classes": full_train.classes,
        "class_to_idx": full_train.class_to_idx,
        "train_samples": len(full_train),
        "val_samples": len(full_val),
        "effective_train_samples": len(train_dataset),
        "effective_val_samples": len(val_dataset),
    }
    write_json(Path(args.output_dir) / "classes.json", class_payload)

    train_loader = make_loader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        drop_last=args.drop_last,
    )
    val_loader = make_loader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        drop_last=False,
    )

    optimizer = optim.AdamW(
        image_sae.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(args.beta1, args.beta2),
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.eta_min)
    scaler = make_grad_scaler(device, args.amp)

    start_epoch = 0
    best_score = float("inf")
    best_epoch = -1
    epochs_since_improvement = 0
    if args.resume:
        start_epoch, best_score, best_epoch, epochs_since_improvement = load_resume_state(
            Path(args.resume),
            image_sae,
            optimizer,
            scheduler,
            scaler,
            device,
            args.resume_model_only,
        )

    metrics_jsonl = Path(args.output_dir) / "metrics" / "metrics.jsonl"
    metrics_json = Path(args.output_dir) / "metrics" / "metrics.json"
    history = []

    print(
        "[Train] "
        f"epochs={args.epochs}, batch_size={args.batch_size}, k={args.img_k_sparse}, "
        f"amp={args.amp}, lambda_ue={args.lambda_ue}"
    )
    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.time()
        train_metrics = run_epoch(
            phase="train",
            epoch=epoch,
            loader=train_loader,
            backbone=backbone,
            image_sae=image_sae,
            optimizer=optimizer,
            scaler=scaler,
            args=args,
            device=device,
        )
        val_metrics = run_epoch(
            phase="val",
            epoch=epoch,
            loader=val_loader,
            backbone=backbone,
            image_sae=image_sae,
            optimizer=None,
            scaler=scaler,
            args=args,
            device=device,
        )
        scheduler.step()

        monitor_value = float(val_metrics["img_rec_loss"])
        improved = monitor_value < (best_score - args.min_delta)
        if improved:
            best_score = monitor_value
            best_epoch = epoch
            epochs_since_improvement = 0
        else:
            epochs_since_improvement += 1

        epoch_record = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            "train": train_metrics,
            "val": val_metrics,
            "monitor": "val_img_rec_loss",
            "monitor_value": monitor_value,
            "best_score": best_score,
            "best_epoch": best_epoch,
            "improved": improved,
            "epoch_elapsed_sec": time.time() - epoch_start,
        }
        history.append(epoch_record)
        append_jsonl(metrics_jsonl, epoch_record)
        write_json(metrics_json, {"history": history, "best_epoch": best_epoch, "best_score": best_score})

        ckpt_dir = Path(args.output_dir) / "checkpoints"
        save_checkpoint(
            ckpt_dir / "last_model.pth",
            epoch=epoch,
            image_sae=image_sae,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            metrics=epoch_record,
            args=args,
            class_to_idx=full_train.class_to_idx,
            best_score=best_score,
            best_epoch=best_epoch,
            epochs_since_improvement=epochs_since_improvement,
            is_best=False,
        )
        if improved:
            save_checkpoint(
                ckpt_dir / "best_model.pth",
                epoch=epoch,
                image_sae=image_sae,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                metrics=epoch_record,
                args=args,
                class_to_idx=full_train.class_to_idx,
                best_score=best_score,
                best_epoch=best_epoch,
                epochs_since_improvement=epochs_since_improvement,
                is_best=True,
            )
            print(f"[Checkpoint] best_model.pth updated at epoch {epoch}, val_img_rec_loss={best_score:.6f}")

        print(
            f"[Epoch {epoch}] "
            f"train_rec={train_metrics['img_rec_loss']:.6f} "
            f"val_rec={val_metrics['img_rec_loss']:.6f} "
            f"best={best_score:.6f}@{best_epoch} "
            f"stale={epochs_since_improvement}/{args.patience}"
        )

        if epochs_since_improvement >= args.patience:
            print(f"[EarlyStopping] no improvement for {args.patience} epochs; stopping at epoch {epoch}.")
            break

    summary = {
        "best_epoch": best_epoch,
        "best_score": best_score,
        "best_checkpoint": str(Path(args.output_dir) / "checkpoints" / "best_model.pth"),
        "last_checkpoint": str(Path(args.output_dir) / "checkpoints" / "last_model.pth"),
        "args": vars(args),
        "dataset": class_payload,
        "history": history,
    }
    write_json(Path(args.output_dir) / "training_summary.json", summary)
    print(f"[Done] best_epoch={best_epoch}, best_val_img_rec_loss={best_score:.6f}")
    print(f"[Done] checkpoint: {summary['best_checkpoint']}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ImageNet image-only SAE on frozen CLIP-RN50 features")

    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--resume_model_only", type=str2bool, default=False)
    parser.add_argument("--overwrite", type=str2bool, default=False)

    parser.add_argument("--use_clip", type=str2bool, default=True)
    parser.add_argument("--clip_variant", type=str, default="RN50")
    parser.add_argument("--clip_pretrained", type=str, default="openai")
    parser.add_argument("--finetuned_model_path", type=str, default=None)
    parser.add_argument("--resnet_weights", type=str, default="IMAGENET1K_V2")
    parser.add_argument("--hook_layer", type=str, default="layer4")
    parser.add_argument("--resize_size", type=int, default=256)
    parser.add_argument("--crop_size", type=int, default=224)

    parser.add_argument("--img_input_dim", type=int, default=2048)
    parser.add_argument("--img_hidden_dim", type=int, default=8192)
    parser.add_argument("--img_k_sparse", type=int, default=16)
    parser.add_argument("--use_bias", type=str2bool, default=False)
    parser.add_argument("--activation", type=str, default="relu", choices=["relu", "gelu"])
    parser.add_argument("--initialization", type=str, default="fan_in", choices=["fan_in", "xavier", "normal"])

    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--min_delta", type=float, default=0.0)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--accumulation_steps", type=int, default=1)
    parser.add_argument("--drop_last", type=str2bool, default=True)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--pin_memory", type=str2bool, default=True)
    parser.add_argument("--train_max_samples", type=int, default=0)
    parser.add_argument("--val_max_samples", type=int, default=0)

    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--eta_min", type=float, default=1e-6)
    parser.add_argument("--grad_clip", type=float, default=1.0)

    parser.add_argument("--lambda_ue", type=float, default=0.0)
    parser.add_argument("--ue_even_weight", type=float, default=1.0)

    parser.add_argument("--amp", type=str2bool, default=False)
    parser.add_argument("--amp_dtype", type=str, default="float16", choices=["float16", "bfloat16"])
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)

    args = parse_args_with_config(parser)
    missing_paths = [
        name
        for name in ("data_root", "output_dir")
        if not str(getattr(args, name, "") or "").strip()
    ]
    if missing_paths:
        raise ValueError(
            "The following paths must be provided through --config or command-line "
            "arguments: " + ", ".join(missing_paths)
        )
    args.data_root = str(Path(args.data_root))
    args.output_dir = str(Path(args.output_dir))
    args.resume = str(args.resume) if args.resume else ""
    args.train_max_samples = (
        None if args.train_max_samples is None or args.train_max_samples <= 0 else args.train_max_samples
    )
    args.val_max_samples = (
        None if args.val_max_samples is None or args.val_max_samples <= 0 else args.val_max_samples
    )
    if args.accumulation_steps < 1:
        raise ValueError("--accumulation_steps must be >= 1")
    return args


def main() -> None:
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()
