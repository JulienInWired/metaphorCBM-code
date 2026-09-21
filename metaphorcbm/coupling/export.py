"""Artifact export for the coupling pipeline."""

from __future__ import annotations

from typing import Dict, List, Tuple

import networkx as nx
import numpy as np
import torch

from .common import build_bipartite_graph, dense_row_topk, ensure_dir, save_json
from .feature_extraction import serialize_records


def expand_reduced_coupling(
    reduced_T: np.ndarray,
    img_original_indices: np.ndarray,
    txt_original_indices: np.ndarray,
    full_shape: tuple[int, int],
) -> np.ndarray:
    full_T = np.zeros(full_shape, dtype=np.float32)
    if reduced_T.size == 0:
        return full_T
    img_idx = img_original_indices.astype(np.int64)
    txt_idx = txt_original_indices.astype(np.int64)
    full_T[np.ix_(img_idx, txt_idx)] = np.asarray(reduced_T, dtype=np.float32)
    return full_T


def derive_directional_conditionals(T: np.ndarray, eps: float = 1e-8) -> Tuple[np.ndarray, np.ndarray]:
    T = np.asarray(T, dtype=np.float32)
    if T.ndim != 2:
        raise ValueError(f"Coupling T must be a 2D array, got shape={T.shape}")
    row_sums = np.clip(T.sum(axis=1, keepdims=True), eps, None)
    col_sums = np.clip(T.sum(axis=0, keepdims=True), eps, None)
    P_v_to_t = (T / row_sums).astype(np.float32)
    P_t_to_v = (T / col_sums).T.astype(np.float32)
    return P_v_to_t, P_t_to_v


def validate_directional_conditionals(T: np.ndarray, P_v_to_t: np.ndarray, P_t_to_v: np.ndarray, eps: float = 1e-8) -> None:
    expected_v_to_t, expected_t_to_v = derive_directional_conditionals(T, eps=eps)
    if P_v_to_t.shape != expected_v_to_t.shape:
        raise ValueError(f"P_v_to_t has shape {P_v_to_t.shape}, expected {expected_v_to_t.shape}.")
    if P_t_to_v.shape != expected_t_to_v.shape:
        raise ValueError(f"P_t_to_v has shape {P_t_to_v.shape}, expected {expected_t_to_v.shape}.")
    if not np.allclose(P_v_to_t, expected_v_to_t, rtol=1e-5, atol=1e-6):
        raise ValueError("P_v_to_t is not the row-conditioned distribution derived from T.")
    if not np.allclose(P_t_to_v, expected_t_to_v, rtol=1e-5, atol=1e-6):
        raise ValueError("P_t_to_v is not the column-conditioned distribution derived from T.")


def summarize_dense_coupling_matrix(T: np.ndarray) -> Dict[str, float]:
    T = np.asarray(T, dtype=np.float32)
    if T.size == 0:
        return {"nnz": 0, "density": 0.0, "min": 0.0, "max": 0.0, "sum": 0.0}
    nnz = int(np.count_nonzero(T))
    density = float(nnz / T.size)
    return {
        "nnz": nnz,
        "density": density,
        "min": float(T.min()),
        "max": float(T.max()),
        "sum": float(T.sum()),
    }


def save_artifacts(
    *,
    output_dir: str,
    config: Dict,
    full_coupling_dense: np.ndarray,
    initial_coupling_dense: np.ndarray,
    concept_basis,
    sample_records,
    optimization_trace: Dict,
    model_config: Dict,
    hub_diagnostics: Dict[str, np.ndarray | float],
) -> Dict[str, str]:
    output_path = ensure_dir(output_dir)

    P_v_to_t, P_t_to_v = derive_directional_conditionals(full_coupling_dense)
    validate_directional_conditionals(full_coupling_dense, P_v_to_t, P_t_to_v)

    coupling_path = output_path / "coupling.npz"
    np.savez_compressed(
        coupling_path,
        T=np.asarray(full_coupling_dense, dtype=np.float32),
        init_T=np.asarray(initial_coupling_dense, dtype=np.float32),
        P_v_to_t=P_v_to_t,
        P_t_to_v=P_t_to_v,
        row_marginals=np.asarray(full_coupling_dense.sum(axis=1), dtype=np.float32),
        col_marginals=np.asarray(full_coupling_dense.sum(axis=0), dtype=np.float32),
        basis_img_indices=concept_basis.selection.basis_img_indices.astype(np.int64),
        basis_txt_indices=concept_basis.selection.basis_txt_indices.astype(np.int64),
    )

    stats_path = output_path / "concept_stats.npz"
    np.savez_compressed(
        stats_path,
        B=concept_basis.B.astype(np.float32),
        C_v=concept_basis.C_v.astype(np.float32),
        C_t=concept_basis.C_t.astype(np.float32),
        mu_v=concept_basis.mu_v.astype(np.float32),
        mu_t=concept_basis.mu_t.astype(np.float32),
        img_reliability=concept_basis.selection.img_reliability.astype(np.float32),
        img_concept_specificity=concept_basis.selection.img_concept_specificity.astype(np.float32),
        img_peak_support_counts=concept_basis.selection.img_peak_support_counts.astype(np.int64),
        img_peak_dominant_patch=concept_basis.selection.img_peak_dominant_patch.astype(np.int64),
        img_peak_dominant_mass=concept_basis.selection.img_peak_dominant_mass.astype(np.float32),
        txt_reliability=concept_basis.selection.txt_reliability.astype(np.float32),
        txt_concept_specificity=concept_basis.selection.txt_concept_specificity.astype(np.float32),
        txt_peak_support_counts=concept_basis.selection.txt_peak_support_counts.astype(np.int64),
        txt_peak_consistency=concept_basis.selection.txt_peak_consistency.astype(np.float32),
        txt_peak_dominant_word=np.asarray(concept_basis.selection.txt_peak_dominant_word, dtype="<U128"),
        txt_peak_dominant_mass=concept_basis.selection.txt_peak_dominant_mass.astype(np.float32),
        txt_peak_stopword_mass=concept_basis.selection.txt_peak_stopword_mass.astype(np.float32),
        txt_peak_invalid_mass=concept_basis.selection.txt_peak_invalid_mass.astype(np.float32),
        img_frequency=concept_basis.selection.img_frequency.astype(np.float32),
        txt_frequency=concept_basis.selection.txt_frequency.astype(np.float32),
        img_support_counts=concept_basis.selection.img_support_counts.astype(np.int64),
        txt_support_counts=concept_basis.selection.txt_support_counts.astype(np.int64),
        img_prefilter_keep_mask_full=np.asarray(concept_basis.image_prefilter.keep_mask, dtype=bool),
        img_prefilter_reason_full=np.asarray(concept_basis.image_prefilter.filter_reason, dtype="<U32"),
        img_peak_support_counts_full=np.asarray(concept_basis.image_prefilter.peak_support_counts, dtype=np.int64),
        img_peak_dominant_patch_full=np.asarray(concept_basis.image_prefilter.peak_dominant_patch, dtype=np.int64),
        img_peak_dominant_mass_full=np.asarray(concept_basis.image_prefilter.peak_dominant_mass, dtype=np.float32),
        txt_prefilter_keep_mask_full=np.asarray(concept_basis.text_prefilter.keep_mask, dtype=bool),
        txt_prefilter_reason_full=np.asarray(concept_basis.text_prefilter.filter_reason, dtype="<U32"),
        txt_peak_support_counts_full=np.asarray(concept_basis.text_prefilter.peak_support_counts, dtype=np.int64),
        txt_peak_consistency_full=np.asarray(concept_basis.text_prefilter.peak_consistency, dtype=np.float32),
        txt_peak_dominant_word_full=np.asarray(concept_basis.text_prefilter.peak_dominant_word, dtype="<U128"),
        txt_peak_dominant_mass_full=np.asarray(concept_basis.text_prefilter.peak_dominant_mass, dtype=np.float32),
        txt_peak_stopword_mass_full=np.asarray(concept_basis.text_prefilter.peak_stopword_mass, dtype=np.float32),
        txt_peak_invalid_mass_full=np.asarray(concept_basis.text_prefilter.peak_invalid_mass, dtype=np.float32),
        hub_nu_v=np.asarray(hub_diagnostics.get("nu_v", []), dtype=np.float32),
        hub_nu_t=np.asarray(hub_diagnostics.get("nu_t", []), dtype=np.float32),
        hub_psi_t_base=np.asarray(hub_diagnostics.get("psi_t_base", hub_diagnostics.get("psi_t", [])), dtype=np.float32),
        hub_psi_t=np.asarray(hub_diagnostics.get("psi_t", []), dtype=np.float32),
        hub_psi_v=np.asarray(hub_diagnostics.get("psi_v", []), dtype=np.float32),
        text_family_ids=np.asarray(hub_diagnostics.get("text_family_ids", []), dtype=np.int64),
        text_family_names=np.asarray(hub_diagnostics.get("text_family_names", []), dtype="<U128"),
        text_family_base_target=np.asarray(hub_diagnostics.get("text_family_base_target", []), dtype=np.float32),
        text_family_target=np.asarray(hub_diagnostics.get("text_family_target", []), dtype=np.float32),
        text_family_effective_size=np.asarray(hub_diagnostics.get("text_family_effective_size", []), dtype=np.float32),
        text_family_hub_final=np.asarray(hub_diagnostics.get("text_family_hub_final", []), dtype=np.float32),
        text_family_ratio_final=np.asarray(hub_diagnostics.get("text_family_ratio_final", []), dtype=np.float32),
        text_hub_raw_final=np.asarray(hub_diagnostics.get("text_hub_raw_final", []), dtype=np.float32),
        text_hub_raw_ratio_final=np.asarray(hub_diagnostics.get("text_hub_raw_ratio_final", []), dtype=np.float32),
        text_hub_ema_final=np.asarray(hub_diagnostics.get("text_hub_ema_final", []), dtype=np.float32),
        text_hub_gain_final=np.asarray(hub_diagnostics.get("text_hub_gain_final", []), dtype=np.float32),
        text_hub_ratio_final=np.asarray(hub_diagnostics.get("text_hub_ratio_final", []), dtype=np.float32),
        image_hub_raw_final=np.asarray(hub_diagnostics.get("image_hub_raw_final", []), dtype=np.float32),
        image_hub_raw_ratio_final=np.asarray(hub_diagnostics.get("image_hub_raw_ratio_final", []), dtype=np.float32),
        image_hub_ema_final=np.asarray(hub_diagnostics.get("image_hub_ema_final", []), dtype=np.float32),
        image_hub_gain_final=np.asarray(hub_diagnostics.get("image_hub_gain_final", []), dtype=np.float32),
        image_hub_ratio_final=np.asarray(hub_diagnostics.get("image_hub_ratio_final", []), dtype=np.float32),
        hub_raw_final=np.asarray(hub_diagnostics.get("hub_raw_final", []), dtype=np.float32),
        hub_raw_ratio_final=np.asarray(hub_diagnostics.get("hub_raw_ratio_final", []), dtype=np.float32),
        hub_ema_final=np.asarray(hub_diagnostics.get("hub_ema_final", []), dtype=np.float32),
        hub_gain_final=np.asarray(hub_diagnostics.get("hub_gain_final", []), dtype=np.float32),
        hub_ratio_final=np.asarray(hub_diagnostics.get("hub_ratio_final", []), dtype=np.float32),
        hub_feedback_warmup_final=np.asarray([hub_diagnostics.get("hub_feedback_warmup_final", 0.0)], dtype=np.float32),
        basis_img_indices=concept_basis.selection.basis_img_indices.astype(np.int64),
        basis_txt_indices=concept_basis.selection.basis_txt_indices.astype(np.int64),
    )

    image_prefilter_path = output_path / "image_concept_filters.json"
    save_json(
        image_prefilter_path,
        {
            "image_concepts": [
                {
                    "concept_id": int(concept_id),
                    "keep": bool(keep),
                    "reason": str(reason),
                    "peak_support": int(peak_support),
                    "dominant_patch": int(dominant_patch),
                    "dominant_patch_mass": float(dominant_mass),
                }
                for concept_id, keep, reason, peak_support, dominant_patch, dominant_mass in zip(
                    range(len(concept_basis.image_prefilter.keep_mask)),
                    concept_basis.image_prefilter.keep_mask.tolist(),
                    concept_basis.image_prefilter.filter_reason.tolist(),
                    concept_basis.image_prefilter.peak_support_counts.tolist(),
                    concept_basis.image_prefilter.peak_dominant_patch.tolist(),
                    concept_basis.image_prefilter.peak_dominant_mass.tolist(),
                )
            ]
        },
    )

    text_prefilter_path = output_path / "text_concept_filters.json"
    save_json(
        text_prefilter_path,
        {
            "text_concepts": [
                {
                    "concept_id": int(concept_id),
                    "keep": bool(keep),
                    "reason": str(reason),
                    "peak_support": int(peak_support),
                    "peak_consistency": float(peak_consistency),
                    "dominant_word": str(dominant_word),
                    "dominant_mass": float(dominant_mass),
                    "stopword_mass": float(stopword_mass),
                    "invalid_mass": float(invalid_mass),
                }
                for concept_id, keep, reason, peak_support, peak_consistency, dominant_word, dominant_mass, stopword_mass, invalid_mass in zip(
                    range(len(concept_basis.text_prefilter.keep_mask)),
                    concept_basis.text_prefilter.keep_mask.tolist(),
                    concept_basis.text_prefilter.filter_reason.tolist(),
                    concept_basis.text_prefilter.peak_support_counts.tolist(),
                    concept_basis.text_prefilter.peak_consistency.tolist(),
                    concept_basis.text_prefilter.peak_dominant_word.tolist(),
                    concept_basis.text_prefilter.peak_dominant_mass.tolist(),
                    concept_basis.text_prefilter.peak_stopword_mass.tolist(),
                    concept_basis.text_prefilter.peak_invalid_mass.tolist(),
                )
            ]
        },
    )

    if config.get("save_sample_cache", False):
        cache_path = output_path / "sample_cache.pt"
        torch.save(serialize_records(sample_records), cache_path)
    else:
        cache_path = None

    graph_view = dense_row_topk(
        full_coupling_dense,
        k=int(config.get("final_row_topk", 0)),
        threshold=float(config.get("final_threshold", 0.0)),
    )
    graph = build_bipartite_graph(graph_view)
    graph_path = output_path / "graph_concepts.graphml"
    nx.write_graphml(graph, graph_path)

    trace_path = output_path / "optimization_trace.json"
    save_json(trace_path, optimization_trace)

    text_family_ratio = np.asarray(hub_diagnostics.get("text_family_ratio_final", []), dtype=np.float32)
    text_family_target = np.asarray(hub_diagnostics.get("text_family_target", []), dtype=np.float32)
    text_family_summary = {
        "enabled": bool(config.get("family_target_enabled", False)),
        "compression": float(config.get("family_target_compression", 0.0)),
        "family_count": int(text_family_target.size),
    }
    if text_family_ratio.size > 0:
        text_family_summary.update(
            {
                "ratio_min_final": float(np.min(text_family_ratio)),
                "ratio_max_final": float(np.max(text_family_ratio)),
            }
        )

    metadata = {
        "config": config,
        "model_config": model_config,
        "coupling_stats": summarize_dense_coupling_matrix(full_coupling_dense),
        "basis_dimensions": {
            "image": int(len(concept_basis.selection.basis_img_indices)),
            "text": int(len(concept_basis.selection.basis_txt_indices)),
        },
        "hubness": {
            "warmup_final": float(hub_diagnostics.get("hub_feedback_warmup_final", 0.0)),
            "ratio_min_final": float(np.min(np.asarray(hub_diagnostics.get("hub_ratio_final", [0.0]), dtype=np.float32))),
            "ratio_max_final": float(np.max(np.asarray(hub_diagnostics.get("hub_ratio_final", [0.0]), dtype=np.float32))),
            "text": {
                "ratio_min_final": float(np.min(np.asarray(hub_diagnostics.get("text_hub_ratio_final", [0.0]), dtype=np.float32))),
                "ratio_max_final": float(np.max(np.asarray(hub_diagnostics.get("text_hub_ratio_final", [0.0]), dtype=np.float32))),
            },
            "text_family": text_family_summary,
            "image": {
                "ratio_min_final": float(np.min(np.asarray(hub_diagnostics.get("image_hub_ratio_final", [0.0]), dtype=np.float32))),
                "ratio_max_final": float(np.max(np.asarray(hub_diagnostics.get("image_hub_ratio_final", [0.0]), dtype=np.float32))),
            },
        },
        "sample_count": int(len(sample_records)),
        "artifacts": {
            "coupling": coupling_path.name,
            "concept_stats": stats_path.name,
            "sample_cache": cache_path.name if cache_path is not None else None,
            "graph": graph_path.name,
            "trace": trace_path.name,
            "image_concept_filters": image_prefilter_path.name,
            "text_concept_filters": text_prefilter_path.name,
        },
    }
    metadata_path = output_path / "metadata.json"
    save_json(metadata_path, metadata)

    return {
        "coupling": str(coupling_path),
        "concept_stats": str(stats_path),
        "sample_cache": str(cache_path) if cache_path is not None else "",
        "graph": str(graph_path),
        "metadata": str(metadata_path),
        "trace": str(trace_path),
        "image_concept_filters": str(image_prefilter_path),
        "text_concept_filters": str(text_prefilter_path),
    }
