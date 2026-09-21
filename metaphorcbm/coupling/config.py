"""Configuration handling for the coupling pipeline."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import torch


COUPLING_DIR = Path(__file__).resolve().parent
PACKAGE_ROOT = COUPLING_DIR.parent
REPOSITORY_ROOT = PACKAGE_ROOT.parent

DEFAULT_SPLIT = "train"
DEFAULT_FINETUNED_MODEL_PATH = None


@dataclass
class CouplingRunConfig:
    checkpoint: str = ""
    output: str = ""
    text_backbone_weights: Optional[str] = None
    dataset: str = "coco"
    data_root: str = ""
    ann_file: Optional[str] = None
    split: str = DEFAULT_SPLIT
    max_samples: Optional[int] = 12000
    caption_mode: str = "all"
    balanced_sampling: bool = False
    batch_size: int = 64
    optimization_batch_size: int = 64
    num_workers: int = 2
    seed: int = 42
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")

    use_clip_backbone: bool = True
    clip_variant: str = "RN50"
    clip_pretrained: str = "openai"
    finetuned_model_path: Optional[str] = DEFAULT_FINETUNED_MODEL_PATH

    img_input_dim: Optional[int] = None
    img_hidden_dim: Optional[int] = None
    img_k_sparse: Optional[int] = None
    d_model: Optional[int] = None
    txt_hidden_dim: Optional[int] = None
    txt_k_sparse: Optional[int] = None
    n_heads: Optional[int] = None
    n_layers: Optional[int] = None
    max_length: Optional[int] = None
    use_bias: Optional[bool] = None
    activation: Optional[str] = None

    activation_power: float = 0.5
    reliability_eps: float = 1e-6
    image_prefilter_enabled: bool = True
    image_prefilter_min_peak_support: int = 32
    image_prefilter_dom_patch_threshold: float = 0.8
    text_prefilter_enabled: bool = True
    text_prefilter_remove_stopwords: bool = True
    text_prefilter_remove_inconsistent: bool = True
    text_prefilter_stop_min_support: int = 1
    text_prefilter_stop_dom_threshold: float = 0.5
    text_prefilter_stop_mass_threshold: float = 0.7
    text_prefilter_consistency_min_support: int = 8
    text_prefilter_p_threshold: float = 0.25
    text_prefilter_dom_content_threshold: float = 0.5

    local_topk: int = 64
    local_steps: int = 25
    local_lr: float = 0.2
    local_gw_weight: float = 1.0
    local_unary_weight: float = 1.0
    local_marginal_weight: float = 0.1
    local_prior_weight: float = 2.0
    local_prior_smoothing: float = 1e-6

    outer_iterations: int = 1
    global_steps: int = 3
    global_lr: float = 0.1
    global_fit_weight: float = 4.0
    global_gw_weight: float = 1.0
    global_marginal_weight_v: float = 1.0
    global_marginal_weight_t: float = 1.0
    global_unary_weight: float = 1.0
    global_unary_warmup_steps: Optional[int] = None
    global_unary_warmup_rounds: int = 2
    init_temperature: float = 0.2
    global_hub_weight: float = 2.0
    global_image_hub_weight: float = 8.0
    hub_target_power: float = 1.0
    family_target_enabled: bool = False
    family_target_compression: float = 0.0
    hub_row_active_tau: float = 1e-6
    hub_ema_decay: float = 0.9
    hub_feedback_gamma: float = 4.0
    hub_gain_min: float = 0.25
    hub_gain_max: float = 4.0
    hub_warmup_start_steps: int = 0
    hub_warmup_ramp_steps: int = 25
    hub_eps: float = 1e-8

    final_row_topk: int = 8
    final_threshold: float = 0.0
    save_sample_cache: bool = False
    cache_precision: str = "float32"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json(cls, path: str) -> "CouplingRunConfig":
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        return cls(**payload)


DEFAULTS = CouplingRunConfig()


def resolve_project_path(path: Optional[str]) -> Optional[str]:
    if path is None:
        return None
    path = str(path).strip()
    if not path:
        return path
    candidate = Path(path)
    if candidate.is_absolute():
        return str(candidate)
    return str((REPOSITORY_ROOT / candidate).resolve())


def normalize_paths(payload: Dict[str, Any]) -> Dict[str, Any]:
    normalized = dict(payload)
    for key in ["checkpoint", "output", "text_backbone_weights", "data_root", "ann_file", "finetuned_model_path"]:
        if key in normalized:
            normalized[key] = resolve_project_path(normalized[key])
    return normalized


def derive_ann_file(data_root: str, split: str) -> str:
    split = split.lower()
    if split not in {"train", "val"}:
        raise ValueError("split must be 'train' or 'val'")
    return str(Path(data_root) / "annotations" / f"captions_{split}2017.json")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Cross-modal coupling training with alternating local/global optimization.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=str, default=None, help="Optional JSON config file.")
    parser.add_argument("--checkpoint", type=str, default=DEFAULTS.checkpoint, help="Path to the SAE checkpoint.")
    parser.add_argument("--output", type=str, default=DEFAULTS.output, help="Output directory.")
    parser.add_argument("--text-backbone-weights", type=str, default=DEFAULTS.text_backbone_weights)
    parser.add_argument("--dataset", type=str, default=DEFAULTS.dataset)
    parser.add_argument("--data-root", type=str, default=DEFAULTS.data_root)
    parser.add_argument("--ann-file", type=str, default=DEFAULTS.ann_file)
    parser.add_argument("--split", type=str, default=DEFAULTS.split, choices=["train", "val"])
    parser.add_argument("--max-samples", type=int, default=DEFAULTS.max_samples)
    parser.add_argument("--caption-mode", type=str, default=DEFAULTS.caption_mode, choices=["first", "all", "random_fixed"])
    parser.add_argument("--balanced-sampling", action="store_true")
    parser.add_argument("--batch-size", type=int, default=DEFAULTS.batch_size)
    parser.add_argument("--optimization-batch-size", type=int, default=DEFAULTS.optimization_batch_size)
    parser.add_argument("--num-workers", type=int, default=DEFAULTS.num_workers)
    parser.add_argument("--seed", type=int, default=DEFAULTS.seed)
    parser.add_argument("--device", type=str, default=DEFAULTS.device)
    parser.add_argument("--use-clip-backbone", action="store_true", default=DEFAULTS.use_clip_backbone)
    parser.add_argument("--no-clip-backbone", dest="use_clip_backbone", action="store_false")
    parser.add_argument("--clip-variant", type=str, default=DEFAULTS.clip_variant)
    parser.add_argument("--clip-pretrained", type=str, default=DEFAULTS.clip_pretrained)
    parser.add_argument("--finetuned-model-path", type=str, default=DEFAULTS.finetuned_model_path)

    parser.add_argument("--img-input-dim", type=int, default=DEFAULTS.img_input_dim)
    parser.add_argument("--img-hidden-dim", type=int, default=DEFAULTS.img_hidden_dim)
    parser.add_argument("--img-k-sparse", type=int, default=DEFAULTS.img_k_sparse)
    parser.add_argument("--d-model", type=int, default=DEFAULTS.d_model)
    parser.add_argument("--txt-hidden-dim", type=int, default=DEFAULTS.txt_hidden_dim)
    parser.add_argument("--txt-k-sparse", type=int, default=DEFAULTS.txt_k_sparse)
    parser.add_argument("--n-heads", type=int, default=DEFAULTS.n_heads)
    parser.add_argument("--n-layers", type=int, default=DEFAULTS.n_layers)
    parser.add_argument("--max-length", type=int, default=DEFAULTS.max_length)
    parser.add_argument("--use-bias", action="store_true", default=DEFAULTS.use_bias)
    parser.add_argument("--no-bias", dest="use_bias", action="store_false")
    parser.set_defaults(use_bias=DEFAULTS.use_bias)
    parser.add_argument(
        "--activation",
        type=str,
        default=DEFAULTS.activation,
        choices=["relu", "gelu"],
    )

    parser.add_argument("--activation-power", type=float, default=DEFAULTS.activation_power)
    parser.add_argument("--reliability-eps", type=float, default=DEFAULTS.reliability_eps)
    parser.add_argument("--no-image-prefilter", dest="image_prefilter_enabled", action="store_false")
    parser.add_argument("--image-prefilter-min-peak-support", type=int, default=DEFAULTS.image_prefilter_min_peak_support)
    parser.add_argument("--image-prefilter-dom-patch-threshold", type=float, default=DEFAULTS.image_prefilter_dom_patch_threshold)
    parser.add_argument("--no-text-prefilter", dest="text_prefilter_enabled", action="store_false")
    parser.add_argument("--no-text-prefilter-stopwords", dest="text_prefilter_remove_stopwords", action="store_false")
    parser.add_argument("--no-text-prefilter-inconsistent", dest="text_prefilter_remove_inconsistent", action="store_false")
    parser.add_argument("--text-prefilter-stop-min-support", type=int, default=DEFAULTS.text_prefilter_stop_min_support)
    parser.add_argument("--text-prefilter-stop-dom-threshold", type=float, default=DEFAULTS.text_prefilter_stop_dom_threshold)
    parser.add_argument("--text-prefilter-stop-mass-threshold", type=float, default=DEFAULTS.text_prefilter_stop_mass_threshold)
    parser.add_argument("--text-prefilter-consistency-min-support", type=int, default=DEFAULTS.text_prefilter_consistency_min_support)
    parser.add_argument("--text-prefilter-p-threshold", type=float, default=DEFAULTS.text_prefilter_p_threshold)
    parser.add_argument("--text-prefilter-dom-content-threshold", type=float, default=DEFAULTS.text_prefilter_dom_content_threshold)
    parser.add_argument("--local-topk", type=int, default=DEFAULTS.local_topk)
    parser.add_argument("--local-steps", type=int, default=DEFAULTS.local_steps)
    parser.add_argument("--local-lr", type=float, default=DEFAULTS.local_lr)
    parser.add_argument("--local-gw-weight", type=float, default=DEFAULTS.local_gw_weight)
    parser.add_argument("--local-unary-weight", type=float, default=DEFAULTS.local_unary_weight)
    parser.add_argument("--local-marginal-weight", type=float, default=DEFAULTS.local_marginal_weight)
    parser.add_argument("--local-prior-weight", type=float, default=DEFAULTS.local_prior_weight)
    parser.add_argument("--local-prior-smoothing", type=float, default=DEFAULTS.local_prior_smoothing)

    parser.add_argument("--outer-iterations", type=int, default=DEFAULTS.outer_iterations)
    parser.add_argument("--global-steps", type=int, default=DEFAULTS.global_steps)
    parser.add_argument("--global-lr", type=float, default=DEFAULTS.global_lr)
    parser.add_argument("--global-fit-weight", type=float, default=DEFAULTS.global_fit_weight)
    parser.add_argument("--global-gw-weight", type=float, default=DEFAULTS.global_gw_weight)
    parser.add_argument("--global-marginal-weight-v", type=float, default=DEFAULTS.global_marginal_weight_v)
    parser.add_argument("--global-marginal-weight-t", type=float, default=DEFAULTS.global_marginal_weight_t)
    parser.add_argument("--global-unary-weight", type=float, default=DEFAULTS.global_unary_weight)
    parser.add_argument("--global-unary-warmup-steps", type=int, default=DEFAULTS.global_unary_warmup_steps)
    parser.add_argument("--global-unary-warmup-rounds", type=int, default=DEFAULTS.global_unary_warmup_rounds)
    parser.add_argument("--init-temperature", type=float, default=DEFAULTS.init_temperature)
    parser.add_argument("--global-hub-weight", type=float, default=DEFAULTS.global_hub_weight)
    parser.add_argument("--global-image-hub-weight", type=float, default=DEFAULTS.global_image_hub_weight)
    parser.add_argument("--hub-target-power", type=float, default=DEFAULTS.hub_target_power)
    parser.add_argument("--family-target-enabled", action="store_true", default=DEFAULTS.family_target_enabled)
    parser.add_argument("--no-family-target", dest="family_target_enabled", action="store_false")
    parser.add_argument("--family-target-compression", type=float, default=DEFAULTS.family_target_compression)
    parser.add_argument("--hub-row-active-tau", type=float, default=DEFAULTS.hub_row_active_tau)
    parser.add_argument("--hub-ema-decay", type=float, default=DEFAULTS.hub_ema_decay)
    parser.add_argument("--hub-feedback-gamma", type=float, default=DEFAULTS.hub_feedback_gamma)
    parser.add_argument("--hub-gain-min", type=float, default=DEFAULTS.hub_gain_min)
    parser.add_argument("--hub-gain-max", type=float, default=DEFAULTS.hub_gain_max)
    parser.add_argument("--hub-warmup-start-steps", type=int, default=DEFAULTS.hub_warmup_start_steps)
    parser.add_argument("--hub-warmup-ramp-steps", type=int, default=DEFAULTS.hub_warmup_ramp_steps)
    parser.add_argument("--hub-eps", type=float, default=DEFAULTS.hub_eps)

    parser.add_argument("--final-row-topk", type=int, default=DEFAULTS.final_row_topk)
    parser.add_argument("--final-threshold", type=float, default=DEFAULTS.final_threshold)
    parser.add_argument("--save-sample-cache", action="store_true", default=DEFAULTS.save_sample_cache)
    parser.add_argument("--no-sample-cache", dest="save_sample_cache", action="store_false")
    parser.add_argument(
        "--cache-precision",
        type=str,
        default=DEFAULTS.cache_precision,
        choices=["float16", "float32"],
    )
    return parser


def config_from_args(args: argparse.Namespace) -> CouplingRunConfig:
    base = {k: v for k, v in vars(args).items() if k != "config"}
    base = normalize_paths(base)
    missing = [
        key
        for key in ("checkpoint", "output", "data_root")
        if not str(base.get(key) or "").strip()
    ]
    if missing:
        raise ValueError(
            "The following paths must be provided through --config or command-line "
            "arguments: " + ", ".join(missing)
        )
    if not base.get("ann_file"):
        base["ann_file"] = derive_ann_file(base["data_root"], base["split"])
    cfg = CouplingRunConfig(**base)
    return cfg
