"""Feature extraction and sparse sample-cache construction."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import scipy.sparse as sp
import torch
from tqdm import tqdm

from .text_utils import INVALID_PEAK_WORD, STOP_WORDS, peak_word_from_token_index


@dataclass
class SampleRecord:
    sample_index: int
    image_id: int
    image_path: str
    caption: str
    img_indices: np.ndarray
    img_values: np.ndarray
    txt_indices: np.ndarray
    txt_values: np.ndarray


@dataclass
class ExtractionResult:
    records: List[SampleRecord]
    img_matrix: sp.csr_matrix
    txt_matrix: sp.csr_matrix
    img_support_counts: np.ndarray
    img_peak_support_counts: np.ndarray
    img_peak_dominant_patch: np.ndarray
    img_peak_dominant_mass: np.ndarray
    txt_support_counts: np.ndarray
    txt_peak_support_counts: np.ndarray
    txt_peak_consistency: np.ndarray
    txt_peak_dominant_word: np.ndarray
    txt_peak_dominant_mass: np.ndarray
    txt_peak_stopword_mass: np.ndarray
    txt_peak_invalid_mass: np.ndarray



def _to_numpy_sparse_vector(x: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
    nz = torch.nonzero(x, as_tuple=False).squeeze(-1)
    if nz.numel() == 0:
        return np.zeros((0,), dtype=np.int64), np.zeros((0,), dtype=np.float32)
    idx = nz.detach().cpu().numpy().astype(np.int64)
    values = x[nz].detach().cpu().numpy().astype(np.float32)
    return idx, values


def _compute_text_peak_statistics(
    *,
    peak_word_counters: List[Counter[str]],
    n_txt: int,
    eps: float = 1e-8,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    peak_support_counts = np.zeros((n_txt,), dtype=np.int64)
    peak_consistency = np.zeros((n_txt,), dtype=np.float64)
    peak_dominant_mass = np.zeros((n_txt,), dtype=np.float64)
    peak_stopword_mass = np.zeros((n_txt,), dtype=np.float64)
    peak_invalid_mass = np.zeros((n_txt,), dtype=np.float64)
    peak_dominant_word = np.full((n_txt,), "", dtype="<U128")

    for concept_id, counter in enumerate(peak_word_counters):
        total = int(sum(counter.values()))
        peak_support_counts[concept_id] = total
        if total <= 0:
            continue

        dominant_word, dominant_count = max(counter.items(), key=lambda item: (item[1], item[0]))
        probs = np.asarray(list(counter.values()), dtype=np.float64) / max(float(total), eps)
        peak_consistency[concept_id] = float(np.square(probs).sum())
        peak_dominant_mass[concept_id] = float(dominant_count / max(float(total), eps))
        peak_stopword_mass[concept_id] = float(
            sum(count for word, count in counter.items() if word in STOP_WORDS) / max(float(total), eps)
        )
        peak_invalid_mass[concept_id] = float(counter.get(INVALID_PEAK_WORD, 0) / max(float(total), eps))
        peak_dominant_word[concept_id] = dominant_word

    return (
        peak_support_counts.astype(np.int64),
        peak_consistency.astype(np.float32),
        peak_dominant_word,
        peak_dominant_mass.astype(np.float32),
        peak_stopword_mass.astype(np.float32),
        peak_invalid_mass.astype(np.float32),
    )


def _compute_image_peak_statistics(
    *,
    peak_patch_counters: List[Counter[int]],
    n_img: int,
    eps: float = 1e-8,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    peak_support_counts = np.zeros((n_img,), dtype=np.int64)
    peak_dominant_patch = np.full((n_img,), -1, dtype=np.int64)
    peak_dominant_mass = np.zeros((n_img,), dtype=np.float64)

    for concept_id, counter in enumerate(peak_patch_counters):
        total = int(sum(counter.values()))
        peak_support_counts[concept_id] = total
        if total <= 0:
            continue

        dominant_patch, dominant_count = max(counter.items(), key=lambda item: (item[1], -item[0]))
        peak_dominant_patch[concept_id] = int(dominant_patch)
        peak_dominant_mass[concept_id] = float(dominant_count / max(float(total), eps))

    return (
        peak_support_counts.astype(np.int64),
        peak_dominant_patch.astype(np.int64),
        peak_dominant_mass.astype(np.float32),
    )


@torch.no_grad()
def extract_sparse_activations(dataloader, models, device: torch.device, logger) -> ExtractionResult:
    image_backbone = models.image_backbone
    image_sae = models.image_sae
    text_backbone = models.text_backbone
    text_sae = models.text_sae

    n_img = int(models.model_config["img_hidden_dim"])
    n_txt = int(models.model_config["txt_hidden_dim"])

    img_rows: List[int] = []
    img_cols: List[int] = []
    img_data: List[float] = []
    txt_rows: List[int] = []
    txt_cols: List[int] = []
    txt_data: List[float] = []

    img_support_counts = np.zeros(n_img, dtype=np.int64)
    txt_support_counts = np.zeros(n_txt, dtype=np.int64)
    text_transforms = getattr(dataloader.dataset, "text_transforms", None)
    tokenizer = getattr(text_transforms, "tokenizer", None)
    if tokenizer is None:
        raise RuntimeError("The coupling dataloader must expose text_transforms.tokenizer for text lexical statistics.")
    special_token_ids = np.asarray(sorted(int(token_id) for token_id in tokenizer.all_special_ids), dtype=np.int64)
    peak_patch_counters: List[Counter[int]] = [Counter() for _ in range(n_img)]
    peak_word_counters: List[Counter[str]] = [Counter() for _ in range(n_txt)]

    records: List[SampleRecord] = []
    sample_index = 0

    for batch in tqdm(dataloader, desc="Extracting sparse activations"):
        batch_size = batch["image"].shape[0]
        images = batch["image"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        image_out = image_backbone(images)
        img_features = image_out["spatial_features"]
        img_sae_out = image_sae(img_features)
        img_global = img_sae_out["global_sparse"]
        img_sparse = img_sae_out["sparse_activations"]
        text_out = text_backbone(input_ids, attention_mask)
        txt_features = text_out["sequence_features"]
        txt_sae_out = text_sae(txt_features, attention_mask)
        txt_global = txt_sae_out["global_sparse"]
        txt_sparse = txt_sae_out["sparse_activations"]

        image_ids = batch["image_id"]
        image_paths = batch["image_path"]
        captions = batch["caption"]

        input_ids_cpu = input_ids.detach().cpu().numpy().astype(np.int64)
        attention_mask_cpu = attention_mask.detach().cpu().numpy().astype(np.int64)

        patch_nonzero = torch.nonzero(img_sparse, as_tuple=False)
        if patch_nonzero.numel() > 0:
            sample_offsets = patch_nonzero[:, 0].detach().cpu().numpy().astype(np.int64)
            patch_positions = patch_nonzero[:, 1].detach().cpu().numpy().astype(np.int64)
            concept_ids = patch_nonzero[:, 2].detach().cpu().numpy().astype(np.int64)
            patch_values = img_sparse[patch_nonzero[:, 0], patch_nonzero[:, 1], patch_nonzero[:, 2]].detach().cpu().numpy().astype(np.float32)
            peak_lookup: dict[tuple[int, int], tuple[int, float]] = {}
            for sample_offset, concept_id, patch_position, patch_value in zip(
                sample_offsets.tolist(),
                concept_ids.tolist(),
                patch_positions.tolist(),
                patch_values.tolist(),
            ):
                key = (sample_offset, concept_id)
                previous = peak_lookup.get(key)
                if previous is None or patch_value > previous[1]:
                    peak_lookup[key] = (patch_position, float(patch_value))
            for (_sample_offset, concept_id), (patch_position, _peak_value) in peak_lookup.items():
                peak_patch_counters[int(concept_id)][int(patch_position)] += 1

        token_nonzero = torch.nonzero(txt_sparse, as_tuple=False)
        if token_nonzero.numel() > 0:
            valid_positions = attention_mask[token_nonzero[:, 0], token_nonzero[:, 1]] > 0
            token_nonzero = token_nonzero[valid_positions]
        if token_nonzero.numel() > 0:
            sample_offsets = token_nonzero[:, 0].detach().cpu().numpy().astype(np.int64)
            token_positions = token_nonzero[:, 1].detach().cpu().numpy().astype(np.int64)
            token_ids = input_ids[token_nonzero[:, 0], token_nonzero[:, 1]].detach().cpu().numpy().astype(np.int64)
            concept_ids = token_nonzero[:, 2].detach().cpu().numpy().astype(np.int64)
            token_values = txt_sparse[token_nonzero[:, 0], token_nonzero[:, 1], token_nonzero[:, 2]].detach().cpu().numpy().astype(np.float32)
            if special_token_ids.size > 0:
                keep = ~np.isin(token_ids, special_token_ids)
                sample_offsets = sample_offsets[keep]
                token_positions = token_positions[keep]
                token_ids = token_ids[keep]
                concept_ids = concept_ids[keep]
                token_values = token_values[keep]
            if token_values.size > 0:
                peak_lookup: dict[tuple[int, int], tuple[int, float]] = {}
                for sample_offset, concept_id, token_position, token_value in zip(
                    sample_offsets.tolist(),
                    concept_ids.tolist(),
                    token_positions.tolist(),
                    token_values.tolist(),
                ):
                    key = (sample_offset, concept_id)
                    previous = peak_lookup.get(key)
                    if previous is None or token_value > previous[1]:
                        peak_lookup[key] = (token_position, float(token_value))
                for (sample_offset, concept_id), (token_position, _peak_value) in peak_lookup.items():
                    normalized_word, _raw_word = peak_word_from_token_index(
                        tokenizer,
                        input_ids_cpu[sample_offset],
                        attention_mask_cpu[sample_offset],
                        special_token_ids,
                        token_position,
                    )
                    peak_word_counters[int(concept_id)][normalized_word] += 1

        for offset in range(batch_size):
            img_idx, img_val = _to_numpy_sparse_vector(img_global[offset])
            txt_idx, txt_val = _to_numpy_sparse_vector(txt_global[offset])

            img_rows.extend([sample_index] * len(img_idx))
            img_cols.extend(img_idx.tolist())
            img_data.extend(img_val.tolist())
            txt_rows.extend([sample_index] * len(txt_idx))
            txt_cols.extend(txt_idx.tolist())
            txt_data.extend(txt_val.tolist())

            if len(img_idx) > 0:
                img_support_counts[img_idx] += 1
            if len(txt_idx) > 0:
                txt_support_counts[txt_idx] += 1

            image_id = image_ids[offset].item() if torch.is_tensor(image_ids[offset]) else int(image_ids[offset])
            image_path = image_paths[offset]
            caption = captions[offset]
            if isinstance(image_path, bytes):
                image_path = image_path.decode("utf-8")
            if isinstance(caption, bytes):
                caption = caption.decode("utf-8")
            records.append(
                SampleRecord(
                    sample_index=sample_index,
                    image_id=int(image_id),
                    image_path=str(image_path),
                    caption=str(caption),
                    img_indices=img_idx,
                    img_values=img_val,
                    txt_indices=txt_idx,
                    txt_values=txt_val,
                )
            )
            sample_index += 1

    img_matrix = sp.csr_matrix((np.asarray(img_data, dtype=np.float32), (np.asarray(img_rows, dtype=np.int64), np.asarray(img_cols, dtype=np.int64))), shape=(sample_index, n_img), dtype=np.float32)
    txt_matrix = sp.csr_matrix((np.asarray(txt_data, dtype=np.float32), (np.asarray(txt_rows, dtype=np.int64), np.asarray(txt_cols, dtype=np.int64))), shape=(sample_index, n_txt), dtype=np.float32)
    (
        img_peak_support_counts,
        img_peak_dominant_patch,
        img_peak_dominant_mass,
    ) = _compute_image_peak_statistics(
        peak_patch_counters=peak_patch_counters,
        n_img=n_img,
    )
    (
        txt_peak_support_counts,
        txt_peak_consistency,
        txt_peak_dominant_word,
        txt_peak_dominant_mass,
        txt_peak_stopword_mass,
        txt_peak_invalid_mass,
    ) = _compute_text_peak_statistics(
        peak_word_counters=peak_word_counters,
        n_txt=n_txt,
    )

    logger.info("Sparse activation extraction complete: %d samples, image nnz=%d, text nnz=%d", sample_index, img_matrix.nnz, txt_matrix.nnz)
    active_img_peaks = img_peak_support_counts[img_peak_support_counts > 0]
    if active_img_peaks.size > 0:
        logger.info(
            "Image peak statistics ready: active concepts=%d median_support=%.1f median_dom_patch_mass=%.4f",
            int(active_img_peaks.size),
            float(np.median(active_img_peaks)),
            float(np.median(img_peak_dominant_mass[img_peak_support_counts > 0])),
        )
    active_peaks = txt_peak_support_counts[txt_peak_support_counts > 0]
    if active_peaks.size > 0:
        logger.info(
            "Text peak statistics ready: active concepts=%d median_support=%.1f median_p=%.4f",
            int(active_peaks.size),
            float(np.median(active_peaks)),
            float(np.median(txt_peak_consistency[txt_peak_support_counts > 0])),
        )
    return ExtractionResult(
        records=records,
        img_matrix=img_matrix,
        txt_matrix=txt_matrix,
        img_support_counts=img_support_counts,
        img_peak_support_counts=img_peak_support_counts,
        img_peak_dominant_patch=img_peak_dominant_patch,
        img_peak_dominant_mass=img_peak_dominant_mass,
        txt_support_counts=txt_support_counts,
        txt_peak_support_counts=txt_peak_support_counts,
        txt_peak_consistency=txt_peak_consistency,
        txt_peak_dominant_word=txt_peak_dominant_word,
        txt_peak_dominant_mass=txt_peak_dominant_mass,
        txt_peak_stopword_mass=txt_peak_stopword_mass,
        txt_peak_invalid_mass=txt_peak_invalid_mass,
    )



def serialize_records(records: List[SampleRecord]) -> Dict[str, object]:
    return {
        "records": [
            {
                "sample_index": record.sample_index,
                "image_id": record.image_id,
                "image_path": record.image_path,
                "caption": record.caption,
                "img_indices": record.img_indices,
                "img_values": record.img_values,
                "txt_indices": record.txt_indices,
                "txt_values": record.txt_values,
            }
            for record in records
        ]
    }
