"""Common utilities for cross-modal coupling."""

from __future__ import annotations

import json
import logging
import math
import os
import random
from pathlib import Path
from typing import Any, Dict, Sequence

import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class NumpyJSONEncoder(json.JSONEncoder):
    def default(self, obj: Any) -> Any:  # noqa: D401
        if isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, Path):
            return str(obj)
        return super().default(obj)


def save_json(path: os.PathLike[str] | str, data: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, cls=NumpyJSONEncoder)


def ensure_dir(path: os.PathLike[str] | str) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def configure_logger(output_dir: os.PathLike[str] | str, name: str = "coupling") -> logging.Logger:
    output_dir = ensure_dir(output_dir)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler = logging.FileHandler(output_dir / f"{name}.log")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    logger.propagate = False
    return logger


def inverse_softplus(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    x = torch.clamp(x, min=eps)
    # Use an exact inverse parameterization so repeated re-wrapping does not
    # inject artificial mass into large dense couplings.
    return x + torch.log(torch.clamp(-torch.expm1(-x), min=eps))


def generalized_kl_torch(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    x_safe = torch.clamp(x, min=0.0)
    y_safe = torch.clamp(y, min=0.0)
    return torch.sum(x_safe * (torch.log(x_safe + eps) - torch.log(y_safe + eps)) - x_safe + y_safe)


def cosine_relation_matrix(weights: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    norms = np.linalg.norm(weights, axis=1, keepdims=True)
    normalized = weights / np.clip(norms, eps, None)
    cosine = normalized @ normalized.T
    cosine = np.clip(cosine, -1.0, 1.0)
    relation = 1.0 - cosine
    np.fill_diagonal(relation, 0.0)
    return relation.astype(np.float32)


def sparse_column_l2_norms(x: sp.csr_matrix, eps: float = 1e-8) -> np.ndarray:
    sq = x.copy()
    sq.data **= 2
    norms = np.sqrt(np.asarray(sq.sum(axis=0)).reshape(-1))
    return np.clip(norms, eps, None)


def sparse_row_topk(matrix: sp.csr_matrix, k: int, threshold: float = 0.0) -> sp.csr_matrix:
    matrix = matrix.tocsr()
    if k <= 0:
        return sp.csr_matrix(matrix.shape, dtype=matrix.dtype)
    data = []
    indices = []
    indptr = [0]
    for row in range(matrix.shape[0]):
        start, end = matrix.indptr[row], matrix.indptr[row + 1]
        cols = matrix.indices[start:end]
        vals = matrix.data[start:end]
        if vals.size == 0:
            indptr.append(len(data))
            continue
        if threshold > 0:
            mask = vals > threshold
            cols = cols[mask]
            vals = vals[mask]
        if vals.size > k:
            keep = np.argpartition(vals, -k)[-k:]
            cols = cols[keep]
            vals = vals[keep]
            order = np.argsort(vals)[::-1]
            cols = cols[order]
            vals = vals[order]
        data.extend(vals.tolist())
        indices.extend(cols.tolist())
        indptr.append(len(data))
    return sp.csr_matrix((np.asarray(data, dtype=matrix.dtype), np.asarray(indices, dtype=np.int64), np.asarray(indptr, dtype=np.int64)), shape=matrix.shape)


def dense_row_topk(matrix: np.ndarray, k: int, threshold: float = 0.0) -> sp.csr_matrix:
    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError(f"dense_row_topk expects a 2D array, got shape={matrix.shape}")
    if k <= 0 or matrix.size == 0:
        return sp.csr_matrix(matrix.shape, dtype=np.float32)

    data = []
    indices = []
    indptr = [0]
    for row_idx in range(matrix.shape[0]):
        row = matrix[row_idx]
        if threshold > 0.0:
            candidate_idx = np.flatnonzero(row > threshold)
        else:
            candidate_idx = np.flatnonzero(row != 0.0)
        if candidate_idx.size == 0:
            indptr.append(len(data))
            continue
        candidate_vals = row[candidate_idx]
        if candidate_vals.size > k:
            keep = np.argpartition(candidate_vals, -k)[-k:]
            candidate_idx = candidate_idx[keep]
            candidate_vals = candidate_vals[keep]
        order = np.argsort(candidate_vals)[::-1]
        candidate_idx = candidate_idx[order]
        candidate_vals = candidate_vals[order]
        data.extend(candidate_vals.astype(np.float32).tolist())
        indices.extend(candidate_idx.astype(np.int64).tolist())
        indptr.append(len(data))

    return sp.csr_matrix(
        (
            np.asarray(data, dtype=np.float32),
            np.asarray(indices, dtype=np.int64),
            np.asarray(indptr, dtype=np.int64),
        ),
        shape=matrix.shape,
        dtype=np.float32,
    )


def build_bipartite_graph(coupling_matrix: sp.csr_matrix) -> nx.Graph:
    coupling_matrix = coupling_matrix.tocsr()
    graph = nx.Graph()
    rows, cols = coupling_matrix.nonzero()
    for i in np.unique(rows):
        graph.add_node(f"img_{int(i)}", bipartite="img", modality="image", concept_id=int(i))
    for j in np.unique(cols):
        graph.add_node(f"txt_{int(j)}", bipartite="txt", modality="text", concept_id=int(j))
    for i, j, w in zip(rows, cols, coupling_matrix.data):
        graph.add_edge(f"img_{int(i)}", f"txt_{int(j)}", weight=float(w))
    return graph


def tensorize(array: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(array, dtype=torch.float32, device=device)


def stable_softplus_normalized(param: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    out = F.softplus(param) + eps
    return out / torch.clamp(out.sum(), min=eps)


def summarize_coupling_matrix(coupling_matrix: sp.csr_matrix) -> Dict[str, float]:
    if coupling_matrix.nnz == 0:
        return {"nnz": 0, "density": 0.0, "min": 0.0, "max": 0.0, "sum": 0.0}
    data = coupling_matrix.data
    total = float(data.sum())
    density = float(coupling_matrix.nnz / (coupling_matrix.shape[0] * coupling_matrix.shape[1]))
    return {
        "nnz": int(coupling_matrix.nnz),
        "density": density,
        "min": float(data.min()),
        "max": float(data.max()),
        "sum": total,
    }


def select_topk_indices(values: np.ndarray, k: int) -> np.ndarray:
    if values.size <= k:
        return np.arange(values.size, dtype=np.int64)
    idx = np.argpartition(values, -k)[-k:]
    idx = idx[np.argsort(values[idx])[::-1]]
    return idx.astype(np.int64)


def maybe_to_numpy(x: Any) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def harmonic_mean(a: float, b: float, eps: float = 1e-8) -> float:
    return 2.0 * a * b / max(a + b, eps)


def clipped_prob(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    x = np.clip(x, 0.0, None)
    s = float(x.sum())
    if s <= eps:
        return np.full_like(x, fill_value=1.0 / max(len(x), 1), dtype=np.float32)
    return (x / s).astype(np.float32)


def linear_warmup_factor(step_index: int, start_steps: int, ramp_steps: int) -> float:
    step_index = int(step_index)
    start_steps = max(int(start_steps), 0)
    ramp_steps = max(int(ramp_steps), 0)
    if step_index < start_steps:
        return 0.0
    if ramp_steps == 0:
        return 1.0
    return float(np.clip((step_index - start_steps) / float(ramp_steps), 0.0, 1.0))


def compute_hubness_target(mu_t: np.ndarray, power: float = 1.0, eps: float = 1e-8) -> np.ndarray:
    mu_t = np.asarray(mu_t, dtype=np.float32)
    powered = np.power(np.clip(mu_t, 0.0, None) + float(eps), float(power))
    return clipped_prob(powered, eps=eps)


def compute_family_compressed_target(
    base_target: np.ndarray,
    family_labels: Sequence[str],
    compression: float,
    eps: float = 1e-8,
) -> tuple[np.ndarray, Dict[str, np.ndarray]]:
    base_target = clipped_prob(np.asarray(base_target, dtype=np.float32), eps=eps)
    labels = np.asarray(list(family_labels), dtype=str).reshape(-1)
    if labels.size != base_target.size:
        raise ValueError(
            f"family_labels length ({labels.size}) must match base_target length ({base_target.size})"
        )

    normalized_labels = []
    for label in labels:
        name = str(label).strip().lower()
        if not name or name == "<invalid>":
            name = "<empty_or_invalid>"
        normalized_labels.append(name)
    family_names, family_ids = np.unique(np.asarray(normalized_labels, dtype="<U128"), return_inverse=True)
    family_ids = family_ids.astype(np.int64, copy=False)

    n_families = int(family_names.size)
    family_base_target = np.bincount(family_ids, weights=base_target.astype(np.float64), minlength=n_families)
    family_target_sq = np.bincount(family_ids, weights=np.square(base_target.astype(np.float64)), minlength=n_families)
    family_effective_size = np.square(family_base_target) / np.clip(family_target_sq, float(eps), None)
    family_effective_size = np.maximum(family_effective_size, 1.0)

    compression = max(float(compression), 0.0)
    raw_family_target = family_base_target / np.power(family_effective_size, compression)
    family_target = clipped_prob(raw_family_target.astype(np.float32), eps=eps)

    within_family = base_target.astype(np.float64) / np.clip(family_base_target[family_ids], float(eps), None)
    compressed_target = family_target[family_ids].astype(np.float64) * within_family
    compressed_target = clipped_prob(compressed_target.astype(np.float32), eps=eps)

    diagnostics: Dict[str, np.ndarray] = {
        "family_ids": family_ids.astype(np.int64),
        "family_names": family_names.astype("<U128"),
        "family_base_target": family_base_target.astype(np.float32),
        "family_target": family_target.astype(np.float32),
        "family_effective_size": family_effective_size.astype(np.float32),
    }
    return compressed_target.astype(np.float32), diagnostics


def compute_text_hubness_numpy(
    T: np.ndarray,
    nu_v: np.ndarray,
    row_active_tau: float = 1e-6,
    eps: float = 1e-8,
) -> np.ndarray:
    T = np.asarray(T, dtype=np.float32)
    nu_v = clipped_prob(np.asarray(nu_v, dtype=np.float32), eps=eps)
    if T.ndim != 2:
        raise ValueError(f"T must be 2D, got shape={T.shape}")

    row_mass = np.clip(T.sum(axis=1), 0.0, None)
    row_gate = row_mass / (row_mass + float(row_active_tau))
    row_weights = clipped_prob(nu_v * row_gate.astype(np.float32), eps=eps)
    cond = T / np.clip(row_mass[:, None], eps, None)
    hubness = np.sum(row_weights[:, None] * cond, axis=0, dtype=np.float32)
    return clipped_prob(hubness, eps=eps)


def compute_image_hubness_numpy(
    T: np.ndarray,
    nu_t: np.ndarray,
    col_active_tau: float = 1e-6,
    eps: float = 1e-8,
) -> np.ndarray:
    T = np.asarray(T, dtype=np.float32)
    nu_t = clipped_prob(np.asarray(nu_t, dtype=np.float32), eps=eps)
    if T.ndim != 2:
        raise ValueError(f"T must be 2D, got shape={T.shape}")

    col_mass = np.clip(T.sum(axis=0), 0.0, None)
    col_gate = col_mass / (col_mass + float(col_active_tau))
    col_weights = clipped_prob(nu_t * col_gate.astype(np.float32), eps=eps)
    cond = T / np.clip(col_mass[None, :], eps, None)
    hubness = np.sum(cond * col_weights[None, :], axis=1, dtype=np.float32)
    return clipped_prob(hubness, eps=eps)


def compute_text_hubness_torch(
    T: torch.Tensor,
    nu_v: torch.Tensor,
    row_active_tau: float = 1e-6,
    eps: float = 1e-8,
) -> torch.Tensor:
    if T.ndim != 2:
        raise ValueError(f"T must be 2D, got shape={tuple(T.shape)}")
    row_mass = torch.clamp(T.sum(dim=1), min=0.0)
    row_gate = row_mass / torch.clamp(row_mass + float(row_active_tau), min=float(eps))
    row_weights = torch.clamp(nu_v, min=0.0) * row_gate
    row_weights = row_weights / torch.clamp(row_weights.sum(), min=float(eps))
    cond = T / torch.clamp(row_mass.unsqueeze(1), min=float(eps))
    hubness = torch.sum(row_weights.unsqueeze(1) * cond, dim=0)
    return hubness / torch.clamp(hubness.sum(), min=float(eps))


def compute_image_hubness_torch(
    T: torch.Tensor,
    nu_t: torch.Tensor,
    col_active_tau: float = 1e-6,
    eps: float = 1e-8,
) -> torch.Tensor:
    if T.ndim != 2:
        raise ValueError(f"T must be 2D, got shape={tuple(T.shape)}")
    col_mass = torch.clamp(T.sum(dim=0), min=0.0)
    col_gate = col_mass / torch.clamp(col_mass + float(col_active_tau), min=float(eps))
    col_weights = torch.clamp(nu_t, min=0.0) * col_gate
    col_weights = col_weights / torch.clamp(col_weights.sum(), min=float(eps))
    cond = T / torch.clamp(col_mass.unsqueeze(0), min=float(eps))
    hubness = torch.sum(cond * col_weights.unsqueeze(0), dim=1)
    return hubness / torch.clamp(hubness.sum(), min=float(eps))


def update_hubness_ema(previous: np.ndarray | None, current: np.ndarray, decay: float) -> np.ndarray:
    current = clipped_prob(np.asarray(current, dtype=np.float32))
    if previous is None:
        return current
    decay = float(np.clip(decay, 0.0, 1.0))
    blended = decay * np.asarray(previous, dtype=np.float32) + (1.0 - decay) * current
    return clipped_prob(blended)


def compute_hubness_gains(
    hubness: np.ndarray,
    target: np.ndarray,
    gamma: float,
    gain_min: float,
    gain_max: float,
    eps: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray]:
    hubness = clipped_prob(np.asarray(hubness, dtype=np.float32), eps=eps)
    target = clipped_prob(np.asarray(target, dtype=np.float32), eps=eps)
    ratio = (hubness + float(eps)) / (target + float(eps))
    if gamma <= 0.0:
        gains = np.ones_like(ratio, dtype=np.float32)
    else:
        gains = np.power(ratio, -float(gamma)).astype(np.float32)
    gains = np.clip(gains, float(gain_min), float(gain_max)).astype(np.float32)
    return gains, ratio.astype(np.float32)
