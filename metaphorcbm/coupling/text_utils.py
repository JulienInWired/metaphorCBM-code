"""Token filtering and word recovery at peak concept activations."""

from __future__ import annotations

import re
from typing import Sequence, Tuple

import numpy as np


INVALID_PEAK_WORD = "<invalid>"
STOP_WORDS = {
    "a",
    "an",
    "the",
    "and",
    "or",
    "but",
    "if",
    "while",
    "with",
    "without",
    "of",
    "in",
    "on",
    "at",
    "by",
    "for",
    "from",
    "to",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "has",
    "have",
    "had",
    "do",
    "does",
    "did",
    "this",
    "that",
    "these",
    "those",
    "it",
    "its",
    "as",
    "into",
    "over",
    "under",
    "up",
    "down",
    "near",
    "far",
    "there",
    "here",
    "then",
    "when",
    "who",
    "most",
    "also",
    "just",
    "very",
    "bird",
    "small",
    "sized",
    "colored",
    "all",
    "well",
    "some",
    "compared",
}


def normalize_peak_word(text: str) -> str:
    normalized = re.sub(r"^[^a-z0-9]+|[^a-z0-9]+$", "", text.lower()).strip()
    return normalized or INVALID_PEAK_WORD


def expand_wordpiece_bounds(tokens: Sequence[str], valid_mask: Sequence[bool], token_idx: int) -> Tuple[int, int]:
    left = int(token_idx)
    while left > 0 and bool(valid_mask[left - 1]) and tokens[left].startswith("##"):
        left -= 1

    right = int(token_idx)
    while right + 1 < len(tokens) and bool(valid_mask[right + 1]) and tokens[right + 1].startswith("##"):
        right += 1
    return left, right


def build_valid_token_mask(
    token_ids: Sequence[int],
    attention_mask: Sequence[int],
    special_token_ids: Sequence[int],
) -> np.ndarray:
    token_ids = np.asarray(token_ids, dtype=np.int64)
    attention_mask = np.asarray(attention_mask, dtype=np.int64)
    valid_mask = attention_mask > 0
    if len(special_token_ids) > 0:
        valid_mask &= ~np.isin(token_ids, np.asarray(special_token_ids, dtype=np.int64))
    return valid_mask.astype(bool)

def peak_word_from_token_index(
    tokenizer,
    token_ids: Sequence[int],
    attention_mask: Sequence[int],
    special_token_ids: Sequence[int],
    token_idx: int,
) -> tuple[str, str]:
    token_ids = np.asarray(token_ids, dtype=np.int64)
    valid_mask = build_valid_token_mask(token_ids, attention_mask, special_token_ids)
    tokens = tokenizer.convert_ids_to_tokens(token_ids.tolist())
    left, right = expand_wordpiece_bounds(tokens, valid_mask, int(token_idx))
    word_text = tokenizer.convert_tokens_to_string(tokens[left : right + 1]).strip()
    return normalize_peak_word(word_text), word_text
