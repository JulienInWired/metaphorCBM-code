"""Concept-basis construction for cross-modal coupling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import scipy.sparse as sp
import torch

from .common import cosine_relation_matrix, sparse_column_l2_norms
from .feature_extraction import SampleRecord
from .text_utils import STOP_WORDS


@dataclass
class ConceptSelection:
    basis_img_indices: np.ndarray
    basis_txt_indices: np.ndarray
    img_reliability: np.ndarray
    img_concept_specificity: np.ndarray
    img_peak_support_counts: np.ndarray
    img_peak_dominant_patch: np.ndarray
    img_peak_dominant_mass: np.ndarray
    txt_reliability: np.ndarray
    txt_concept_specificity: np.ndarray
    txt_peak_support_counts: np.ndarray
    txt_peak_consistency: np.ndarray
    txt_peak_dominant_word: np.ndarray
    txt_peak_dominant_mass: np.ndarray
    txt_peak_stopword_mass: np.ndarray
    txt_peak_invalid_mass: np.ndarray
    img_frequency: np.ndarray
    txt_frequency: np.ndarray
    img_support_counts: np.ndarray
    txt_support_counts: np.ndarray


@dataclass
class TextConceptPrefilterDiagnostics:
    peak_support_counts: np.ndarray
    peak_consistency: np.ndarray
    peak_dominant_word: np.ndarray
    peak_dominant_mass: np.ndarray
    peak_stopword_mass: np.ndarray
    peak_invalid_mass: np.ndarray
    keep_mask: np.ndarray
    filter_reason: np.ndarray


@dataclass
class ImageConceptPrefilterDiagnostics:
    peak_support_counts: np.ndarray
    peak_dominant_patch: np.ndarray
    peak_dominant_mass: np.ndarray
    keep_mask: np.ndarray
    filter_reason: np.ndarray


@dataclass
class ReducedSampleRecord:
    sample_index: int
    image_id: int
    image_path: str
    caption: str
    img_idx: np.ndarray
    img_values: np.ndarray
    img_mass: np.ndarray
    txt_idx: np.ndarray
    txt_values: np.ndarray
    txt_mass: np.ndarray


@dataclass
class ConceptBasis:
    selection: ConceptSelection
    image_prefilter: ImageConceptPrefilterDiagnostics
    text_prefilter: TextConceptPrefilterDiagnostics
    reduced_records: List[ReducedSampleRecord]
    C_v: np.ndarray
    C_t: np.ndarray
    B: np.ndarray
    mu_v: np.ndarray
    mu_t: np.ndarray
    decoder_img: np.ndarray
    decoder_txt: np.ndarray


def compute_image_statistics(
    support_counts: np.ndarray,
    num_samples: int,
    eps: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    frequency = support_counts.astype(np.float64) / max(num_samples, 1)
    concept_specificity = np.zeros_like(frequency, dtype=np.float64)
    reliability = np.ones_like(frequency, dtype=np.float64)
    active = support_counts > 0
    if active.any():
        concept_specificity[active] = -np.log(np.clip(frequency[active], eps, 1.0))
    return (
        frequency.astype(np.float32),
        concept_specificity.astype(np.float32),
        reliability.astype(np.float32),
    )


def compute_text_statistics(
    support_counts: np.ndarray,
    num_samples: int,
    eps: float,
) -> Tuple[np.ndarray, np.ndarray]:
    frequency = support_counts.astype(np.float64) / max(num_samples, 1)
    concept_specificity = np.zeros_like(frequency, dtype=np.float64)
    active = support_counts > 0
    if active.any():
        concept_specificity[active] = -np.log(np.clip(frequency[active], eps, 1.0))
    return frequency.astype(np.float32), concept_specificity.astype(np.float32)


def compute_text_prefilter_mask(
    *,
    peak_support_counts: np.ndarray,
    peak_consistency: np.ndarray,
    peak_dominant_word: np.ndarray,
    peak_dominant_mass: np.ndarray,
    peak_stopword_mass: np.ndarray,
    peak_invalid_mass: np.ndarray,
    enabled: bool,
    remove_stopwords: bool,
    remove_inconsistent: bool,
    stop_min_support: int,
    stop_dom_threshold: float,
    stop_mass_threshold: float,
    consistency_min_support: int,
    p_threshold: float,
    dom_content_threshold: float,
) -> Tuple[np.ndarray, np.ndarray]:
    keep_mask = np.ones_like(peak_support_counts, dtype=bool)
    filter_reason = np.full(peak_support_counts.shape, "kept", dtype="<U32")
    if not enabled:
        return keep_mask, filter_reason

    dominant_word = np.asarray(peak_dominant_word, dtype=np.str_)
    dominant_is_stop = np.isin(dominant_word, np.asarray(sorted(STOP_WORDS), dtype=np.str_))
    support = np.asarray(peak_support_counts, dtype=np.int64)

    no_peak_support_mask = keep_mask & (support <= 4)
    keep_mask[no_peak_support_mask] = False
    filter_reason[no_peak_support_mask] = "no_peak_support"

    empty_dominant_mask = keep_mask & (dominant_word == "")
    keep_mask[empty_dominant_mask] = False
    filter_reason[empty_dominant_mask] = "empty_dominant"

    if remove_stopwords:
        dominant_stop_mask = (
            keep_mask
            & (support >= int(stop_min_support))
            & dominant_is_stop
            & (np.asarray(peak_dominant_mass, dtype=np.float32) >= float(stop_dom_threshold))
        )
        keep_mask[dominant_stop_mask] = False
        filter_reason[dominant_stop_mask] = "stopword_dominant"

        stop_mass_mask = (
            keep_mask
            & (support >= int(stop_min_support))
            & (
                np.asarray(peak_stopword_mass, dtype=np.float32)
                + np.asarray(peak_invalid_mass, dtype=np.float32)
                >= float(stop_mass_threshold)
            )
        )
        keep_mask[stop_mass_mask] = False
        filter_reason[stop_mass_mask] = "stopword_mass"

    if remove_inconsistent:
        inconsistent_mask = (
            keep_mask
            & (support >= int(consistency_min_support))
            & (np.asarray(peak_consistency, dtype=np.float32) < float(p_threshold))
            & (np.asarray(peak_dominant_mass, dtype=np.float32) < float(dom_content_threshold))
        )
        keep_mask[inconsistent_mask] = False
        filter_reason[inconsistent_mask] = "inconsistent"

    return keep_mask.astype(bool), filter_reason


def compute_image_prefilter_mask(
    *,
    peak_support_counts: np.ndarray,
    peak_dominant_patch: np.ndarray,
    peak_dominant_mass: np.ndarray,
    enabled: bool,
    min_peak_support: int,
    dom_patch_threshold: float,
) -> Tuple[np.ndarray, np.ndarray]:
    keep_mask = np.ones_like(peak_support_counts, dtype=bool)
    filter_reason = np.full(peak_support_counts.shape, "kept", dtype="<U32")
    if not enabled:
        return keep_mask, filter_reason

    support = np.asarray(peak_support_counts, dtype=np.int64)
    dominant_patch = np.asarray(peak_dominant_patch, dtype=np.int64)
    dominant_mass = np.asarray(peak_dominant_mass, dtype=np.float32)

    spatial_peak_locked_mask = (
        keep_mask
        & (support >= int(min_peak_support))
        & (dominant_patch >= 0)
        & (dominant_mass >= float(dom_patch_threshold))
    )
    keep_mask[spatial_peak_locked_mask] = False
    filter_reason[spatial_peak_locked_mask] = "spatial_peak_locked"
    return keep_mask.astype(bool), filter_reason


def compute_unary_cost(
    img_matrix: sp.csr_matrix,
    txt_matrix: sp.csr_matrix,
    img_indices: np.ndarray,
    txt_indices: np.ndarray,
) -> np.ndarray:
    if img_indices.size == 0 or txt_indices.size == 0:
        return np.zeros((img_indices.size, txt_indices.size), dtype=np.float32)
    img_sub = img_matrix[:, img_indices].tocsr()
    txt_sub = txt_matrix[:, txt_indices].tocsr()
    cross = (img_sub.T @ txt_sub).toarray().astype(np.float32)
    img_norms = sparse_column_l2_norms(img_sub)
    txt_norms = sparse_column_l2_norms(txt_sub)
    denom = np.outer(img_norms, txt_norms)
    cosine = cross / denom
    cosine = np.clip(cosine, -1.0, 1.0)
    return (1.0 - cosine).astype(np.float32)


def compute_decoder_geometry(decoder_weights: torch.Tensor, selected_indices: np.ndarray) -> np.ndarray:
    weights = (
        decoder_weights.detach().cpu().numpy().T
        if decoder_weights.ndim == 2 and decoder_weights.shape[0] < decoder_weights.shape[1]
        else decoder_weights.detach().cpu().numpy()
    )
    weights = weights[selected_indices]
    return cosine_relation_matrix(weights)


def _build_index_map(selected_indices: np.ndarray, full_dim: int) -> np.ndarray:
    index_map = np.full((full_dim,), -1, dtype=np.int64)
    index_map[selected_indices.astype(np.int64)] = np.arange(selected_indices.size, dtype=np.int64)
    return index_map


def _reindex_sparse_values(indices: np.ndarray, values: np.ndarray, index_map: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if indices.size == 0:
        return np.zeros((0,), dtype=np.int64), np.zeros((0,), dtype=np.float32)
    mapped = index_map[indices.astype(np.int64)]
    keep = mapped >= 0
    if not np.any(keep):
        return np.zeros((0,), dtype=np.int64), np.zeros((0,), dtype=np.float32)
    return mapped[keep].astype(np.int64), values[keep].astype(np.float32)


def _build_sample_reduced_record(
    record: SampleRecord,
    img_reliability: np.ndarray,
    txt_reliability: np.ndarray,
    img_index_map: np.ndarray,
    txt_index_map: np.ndarray,
    activation_power: float,
    eps: float = 1e-8,
) -> ReducedSampleRecord:
    img_idx, img_values = _reindex_sparse_values(record.img_indices, record.img_values, img_index_map)
    txt_idx, txt_values = _reindex_sparse_values(record.txt_indices, record.txt_values, txt_index_map)

    if img_idx.size > 0:
        weighted_img = img_reliability[img_idx] * np.power(np.clip(img_values, 0.0, None), activation_power)
        img_mass_sum = float(weighted_img.sum())
        img_mass = (weighted_img / (img_mass_sum + eps)).astype(np.float32) if img_mass_sum > 0.0 else np.zeros_like(img_values, dtype=np.float32)
    else:
        img_mass = np.zeros((0,), dtype=np.float32)

    if txt_idx.size > 0:
        weighted_txt = txt_reliability[txt_idx] * np.power(np.clip(txt_values, 0.0, None), activation_power)
        txt_mass_sum = float(weighted_txt.sum())
        txt_mass = (weighted_txt / (txt_mass_sum + eps)).astype(np.float32) if txt_mass_sum > 0.0 else np.zeros_like(txt_values, dtype=np.float32)
    else:
        txt_mass = np.zeros((0,), dtype=np.float32)

    return ReducedSampleRecord(
        sample_index=record.sample_index,
        image_id=record.image_id,
        image_path=record.image_path,
        caption=record.caption,
        img_idx=img_idx,
        img_values=img_values,
        img_mass=img_mass,
        txt_idx=txt_idx,
        txt_values=txt_values,
        txt_mass=txt_mass,
    )


def compute_global_priors(reduced_records: List[ReducedSampleRecord], n_img: int, n_txt: int) -> Tuple[np.ndarray, np.ndarray]:
    mu_v = np.zeros(n_img, dtype=np.float64)
    mu_t = np.zeros(n_txt, dtype=np.float64)
    total = len(reduced_records)
    for record in reduced_records:
        if record.img_idx.size > 0:
            mu_v[record.img_idx] += record.img_mass.astype(np.float64)
        if record.txt_idx.size > 0:
            mu_t[record.txt_idx] += record.txt_mass.astype(np.float64)
    if total > 0:
        mu_v /= float(total)
        mu_t /= float(total)
    return mu_v.astype(np.float32), mu_t.astype(np.float32)


def build_concept_basis(
    *,
    records: List[SampleRecord],
    img_matrix: sp.csr_matrix,
    txt_matrix: sp.csr_matrix,
    img_support_counts: np.ndarray,
    img_peak_support_counts: np.ndarray,
    img_peak_dominant_patch: np.ndarray,
    img_peak_dominant_mass: np.ndarray,
    txt_support_counts: np.ndarray,
    txt_peak_support_counts: np.ndarray,
    txt_peak_consistency: np.ndarray,
    txt_peak_dominant_word: np.ndarray,
    txt_peak_dominant_mass: np.ndarray,
    txt_peak_stopword_mass: np.ndarray,
    txt_peak_invalid_mass: np.ndarray,
    img_decoder_weights: torch.Tensor,
    txt_decoder_weights: torch.Tensor,
    activation_power: float,
    reliability_eps: float,
    image_prefilter_enabled: bool,
    image_prefilter_min_peak_support: int,
    image_prefilter_dom_patch_threshold: float,
    text_prefilter_enabled: bool,
    text_prefilter_remove_stopwords: bool,
    text_prefilter_remove_inconsistent: bool,
    text_prefilter_stop_min_support: int,
    text_prefilter_stop_dom_threshold: float,
    text_prefilter_stop_mass_threshold: float,
    text_prefilter_consistency_min_support: int,
    text_prefilter_p_threshold: float,
    text_prefilter_dom_content_threshold: float,
) -> ConceptBasis:
    num_samples = len(records)
    img_frequency, img_concept_specificity, img_reliability_full = compute_image_statistics(
        img_support_counts, num_samples, reliability_eps
    )
    txt_frequency, txt_concept_specificity = compute_text_statistics(
        txt_support_counts, num_samples, reliability_eps
    )

    img_keep_mask, img_filter_reason = compute_image_prefilter_mask(
        peak_support_counts=img_peak_support_counts,
        peak_dominant_patch=img_peak_dominant_patch,
        peak_dominant_mass=img_peak_dominant_mass,
        enabled=image_prefilter_enabled,
        min_peak_support=image_prefilter_min_peak_support,
        dom_patch_threshold=image_prefilter_dom_patch_threshold,
    )

    txt_keep_mask, txt_filter_reason = compute_text_prefilter_mask(
        peak_support_counts=txt_peak_support_counts,
        peak_consistency=txt_peak_consistency,
        peak_dominant_word=txt_peak_dominant_word,
        peak_dominant_mass=txt_peak_dominant_mass,
        peak_stopword_mass=txt_peak_stopword_mass,
        peak_invalid_mass=txt_peak_invalid_mass,
        enabled=text_prefilter_enabled,
        remove_stopwords=text_prefilter_remove_stopwords,
        remove_inconsistent=text_prefilter_remove_inconsistent,
        stop_min_support=text_prefilter_stop_min_support,
        stop_dom_threshold=text_prefilter_stop_dom_threshold,
        stop_mass_threshold=text_prefilter_stop_mass_threshold,
        consistency_min_support=text_prefilter_consistency_min_support,
        p_threshold=text_prefilter_p_threshold,
        dom_content_threshold=text_prefilter_dom_content_threshold,
    )

    basis_img_indices = np.flatnonzero(img_keep_mask).astype(np.int64)
    basis_txt_indices = np.flatnonzero(txt_keep_mask).astype(np.int64)
    if basis_img_indices.size == 0:
        raise RuntimeError("Image concept prefilter removed every image concept; coupling cannot proceed.")
    if basis_txt_indices.size == 0:
        raise RuntimeError("Text concept prefilter removed every text concept; coupling cannot proceed.")

    img_reliability = img_reliability_full[basis_img_indices]
    txt_reliability_full = np.ones_like(txt_frequency, dtype=np.float32)
    txt_reliability = txt_reliability_full[basis_txt_indices]

    img_index_map = _build_index_map(basis_img_indices, img_matrix.shape[1])
    txt_index_map = _build_index_map(basis_txt_indices, txt_matrix.shape[1])
    reduced_records = [
        _build_sample_reduced_record(
            record,
            img_reliability,
            txt_reliability,
            img_index_map,
            txt_index_map,
            activation_power,
        )
        for record in records
    ]

    C_v = compute_decoder_geometry(img_decoder_weights, basis_img_indices)
    C_t = compute_decoder_geometry(txt_decoder_weights, basis_txt_indices)
    B = compute_unary_cost(img_matrix, txt_matrix, basis_img_indices, basis_txt_indices)
    mu_v, mu_t = compute_global_priors(reduced_records, len(basis_img_indices), len(basis_txt_indices))

    selection = ConceptSelection(
        basis_img_indices=basis_img_indices,
        basis_txt_indices=basis_txt_indices,
        img_reliability=img_reliability,
        img_concept_specificity=img_concept_specificity[basis_img_indices],
        img_peak_support_counts=img_peak_support_counts[basis_img_indices].astype(np.int64),
        img_peak_dominant_patch=img_peak_dominant_patch[basis_img_indices].astype(np.int64),
        img_peak_dominant_mass=img_peak_dominant_mass[basis_img_indices].astype(np.float32),
        txt_reliability=txt_reliability,
        txt_concept_specificity=txt_concept_specificity[basis_txt_indices],
        txt_peak_support_counts=txt_peak_support_counts[basis_txt_indices].astype(np.int64),
        txt_peak_consistency=txt_peak_consistency[basis_txt_indices].astype(np.float32),
        txt_peak_dominant_word=np.asarray(txt_peak_dominant_word[basis_txt_indices], dtype="<U128"),
        txt_peak_dominant_mass=txt_peak_dominant_mass[basis_txt_indices].astype(np.float32),
        txt_peak_stopword_mass=txt_peak_stopword_mass[basis_txt_indices].astype(np.float32),
        txt_peak_invalid_mass=txt_peak_invalid_mass[basis_txt_indices].astype(np.float32),
        img_frequency=img_frequency[basis_img_indices],
        txt_frequency=txt_frequency[basis_txt_indices],
        img_support_counts=img_support_counts[basis_img_indices],
        txt_support_counts=txt_support_counts[basis_txt_indices],
    )
    image_prefilter = ImageConceptPrefilterDiagnostics(
        peak_support_counts=np.asarray(img_peak_support_counts, dtype=np.int64),
        peak_dominant_patch=np.asarray(img_peak_dominant_patch, dtype=np.int64),
        peak_dominant_mass=np.asarray(img_peak_dominant_mass, dtype=np.float32),
        keep_mask=np.asarray(img_keep_mask, dtype=bool),
        filter_reason=np.asarray(img_filter_reason, dtype="<U32"),
    )
    text_prefilter = TextConceptPrefilterDiagnostics(
        peak_support_counts=np.asarray(txt_peak_support_counts, dtype=np.int64),
        peak_consistency=np.asarray(txt_peak_consistency, dtype=np.float32),
        peak_dominant_word=np.asarray(txt_peak_dominant_word, dtype="<U128"),
        peak_dominant_mass=np.asarray(txt_peak_dominant_mass, dtype=np.float32),
        peak_stopword_mass=np.asarray(txt_peak_stopword_mass, dtype=np.float32),
        peak_invalid_mass=np.asarray(txt_peak_invalid_mass, dtype=np.float32),
        keep_mask=np.asarray(txt_keep_mask, dtype=bool),
        filter_reason=np.asarray(txt_filter_reason, dtype="<U32"),
    )

    decoder_img = img_decoder_weights.detach().cpu().numpy().T[basis_img_indices].astype(np.float32)
    decoder_txt = txt_decoder_weights.detach().cpu().numpy().T[basis_txt_indices].astype(np.float32)

    return ConceptBasis(
        selection=selection,
        image_prefilter=image_prefilter,
        text_prefilter=text_prefilter,
        reduced_records=reduced_records,
        C_v=C_v,
        C_t=C_t,
        B=B,
        mu_v=mu_v.astype(np.float32),
        mu_t=mu_t.astype(np.float32),
        decoder_img=decoder_img,
        decoder_txt=decoder_txt,
    )
