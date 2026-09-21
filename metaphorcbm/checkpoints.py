"""Checkpoint schemas shared by training and inference pipelines."""

from __future__ import annotations

import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Dict, Optional

import torch


JOINT_SAE_CHECKPOINT_TYPE = "metaphorcbm.joint_sae"
JOINT_SAE_CHECKPOINT_VERSION = 1
JOINT_SAE_COMPONENTS = ("image_sae", "text_backbone", "text_sae")


class CheckpointFormatError(RuntimeError):
    """Raised when a checkpoint cannot supply the requested model state."""


def load_checkpoint_file(
    path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> Dict[str, Any]:
    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    try:
        payload = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
    except TypeError:
        payload = torch.load(checkpoint_path, map_location=map_location)
    if not isinstance(payload, Mapping):
        raise CheckpointFormatError(
            f"Checkpoint must contain a mapping: {checkpoint_path}"
        )
    return dict(payload)


def _state_from_container(container: Any, component: str) -> Optional[Dict[str, Any]]:
    if not isinstance(container, Mapping):
        return None
    nested = container.get(component)
    if isinstance(nested, Mapping):
        return nested
    prefix = f"{component}."
    prefixed = {
        str(key)[len(prefix) :]: value
        for key, value in container.items()
        if str(key).startswith(prefix)
    }
    return prefixed or None


def extract_component_state(
    checkpoint: Mapping[str, Any],
    component: str,
) -> Dict[str, Any]:
    """Extract a component from nested, prefixed, or component-specific state dictionaries."""

    state = _state_from_container(checkpoint.get("model_state_dict"), component)
    if state is None:
        state = _state_from_container(checkpoint, component)
    if state is None and component == "image_sae":
        legacy = checkpoint.get("image_sae_state_dict")
        if isinstance(legacy, Mapping):
            state = legacy
    if not state:
        raise CheckpointFormatError(
            f"Checkpoint does not contain a nonempty {component!r} state dictionary."
        )
    return state


def normalize_joint_sae_checkpoint(checkpoint: Mapping[str, Any]) -> Dict[str, Any]:
    """Return a validated joint SAE checkpoint in the canonical nested form."""

    checkpoint_type = checkpoint.get("checkpoint_type")
    if checkpoint_type not in (None, JOINT_SAE_CHECKPOINT_TYPE):
        raise CheckpointFormatError(
            f"Expected checkpoint type {JOINT_SAE_CHECKPOINT_TYPE!r}, "
            f"received {checkpoint_type!r}."
        )
    version = int(checkpoint.get("checkpoint_version", 0))
    if version > JOINT_SAE_CHECKPOINT_VERSION:
        raise CheckpointFormatError(
            f"Checkpoint version {version} is newer than supported version "
            f"{JOINT_SAE_CHECKPOINT_VERSION}."
        )

    normalized = dict(checkpoint)
    normalized["checkpoint_type"] = JOINT_SAE_CHECKPOINT_TYPE
    normalized["checkpoint_version"] = JOINT_SAE_CHECKPOINT_VERSION
    normalized["model_state_dict"] = {
        component: extract_component_state(checkpoint, component)
        for component in JOINT_SAE_COMPONENTS
    }

    if not isinstance(normalized.get("model_config"), Mapping):
        extra_info = checkpoint.get("extra_info")
        if isinstance(extra_info, Mapping) and isinstance(
            extra_info.get("model_config"), Mapping
        ):
            normalized["model_config"] = dict(extra_info["model_config"])
    return normalized


def load_joint_sae_checkpoint(
    path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> Dict[str, Any]:
    return normalize_joint_sae_checkpoint(
        load_checkpoint_file(path, map_location=map_location)
    )


def build_joint_sae_checkpoint(
    *,
    epoch: int,
    model_state: Mapping[str, Mapping[str, Any]],
    model_config: Mapping[str, Any],
    optimizer_state: Mapping[str, Any],
    scheduler_state: Optional[Mapping[str, Any]] = None,
    metrics: Optional[Mapping[str, float]] = None,
    extra_info: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "checkpoint_type": JOINT_SAE_CHECKPOINT_TYPE,
        "checkpoint_version": JOINT_SAE_CHECKPOINT_VERSION,
        "epoch": int(epoch),
        "model_config": dict(model_config),
        "model_state_dict": {
            component: model_state[component]
            for component in JOINT_SAE_COMPONENTS
        },
        "optimizer_state_dict": dict(optimizer_state),
        "metrics": dict(metrics or {}),
        "timestamp": time.time(),
    }
    if scheduler_state is not None:
        payload["scheduler_state_dict"] = dict(scheduler_state)
    if extra_info:
        payload["extra_info"] = dict(extra_info)
    return normalize_joint_sae_checkpoint(payload)


__all__ = [
    "CheckpointFormatError",
    "JOINT_SAE_CHECKPOINT_TYPE",
    "JOINT_SAE_CHECKPOINT_VERSION",
    "build_joint_sae_checkpoint",
    "extract_component_state",
    "load_checkpoint_file",
    "load_joint_sae_checkpoint",
    "normalize_joint_sae_checkpoint",
]
