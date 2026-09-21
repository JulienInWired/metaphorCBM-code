"""Model loading helpers for the coupling pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import torch
import torch.nn as nn

from metaphorcbm.checkpoints import load_joint_sae_checkpoint
from metaphorcbm.data import TextTransforms
from metaphorcbm.models import (
    ImageSAE,
    ResNetWithHooks,
    create_clip_resnet_backbone,
    create_text_sae,
    create_transformer_with_hooks,
)

from .config import CouplingRunConfig


@dataclass
class LoadedModels:
    image_backbone: nn.Module
    image_sae: nn.Module
    text_backbone: nn.Module
    text_sae: nn.Module
    image_preprocess: Any
    model_config: Dict[str, Any]


DEFAULT_MODEL_CONFIG: Dict[str, Any] = {
    "img_input_dim": 2048,
    "img_hidden_dim": 8192,
    "img_k_sparse": 4,
    "d_model": 512,
    "txt_hidden_dim": 4096,
    "txt_k_sparse": 4,
    "n_heads": 8,
    "n_layers": 6,
    "max_length": 40,
    "use_bias": False,
    "activation": "relu",
}


def infer_model_config(checkpoint: Dict[str, Any], cfg: CouplingRunConfig) -> Dict[str, Any]:
    model_config = dict(DEFAULT_MODEL_CONFIG)
    embedded_config = checkpoint.get("model_config")
    if isinstance(embedded_config, dict):
        model_config.update(embedded_config)

    state = checkpoint.get("model_state_dict", {})
    image_state = state["image_sae"]
    text_state = state["text_sae"]

    img_decoder = image_state.get("decoder.0.weight")
    if img_decoder is None:
        img_decoder = image_state.get("decoder.1.weight")
    txt_decoder = text_state.get("decoder.0.weight")
    if txt_decoder is None:
        txt_decoder = text_state.get("decoder.1.weight")

    if img_decoder is not None:
        model_config["img_input_dim"] = int(img_decoder.shape[0])
        model_config["img_hidden_dim"] = int(img_decoder.shape[1])
    if txt_decoder is not None:
        model_config["d_model"] = int(txt_decoder.shape[0])
        model_config["txt_hidden_dim"] = int(txt_decoder.shape[1])

    overrides = {
        "img_input_dim": cfg.img_input_dim,
        "img_hidden_dim": cfg.img_hidden_dim,
        "img_k_sparse": cfg.img_k_sparse,
        "d_model": cfg.d_model,
        "txt_hidden_dim": cfg.txt_hidden_dim,
        "txt_k_sparse": cfg.txt_k_sparse,
        "n_heads": cfg.n_heads,
        "n_layers": cfg.n_layers,
        "max_length": cfg.max_length,
        "use_bias": cfg.use_bias,
        "activation": cfg.activation,
    }
    for key, value in overrides.items():
        if value is not None:
            model_config[key] = value
    return model_config



def load_models(cfg: CouplingRunConfig, device: torch.device, logger) -> LoadedModels:
    checkpoint = load_joint_sae_checkpoint(cfg.checkpoint, map_location="cpu")
    model_config = infer_model_config(checkpoint, cfg)

    logger.info("Loading image backbone and SAE...")
    if cfg.use_clip_backbone:
        finetuned_model_path = cfg.finetuned_model_path
        if isinstance(finetuned_model_path, str) and not finetuned_model_path.strip():
            finetuned_model_path = None
        if finetuned_model_path and not Path(finetuned_model_path).is_file():
            raise FileNotFoundError(
                f"Finetuned CLIP checkpoint not found: {finetuned_model_path}"
            )
        image_backbone = create_clip_resnet_backbone(
            {
                "clip_variant": cfg.clip_variant,
                "pretrained": cfg.clip_pretrained,
                "hook_layer": "layer4",
                "freeze_backbone": True,
                "return_features": True,
                "finetuned_model_path": finetuned_model_path,
            }
        ).to(device).eval()
        image_preprocess = image_backbone.preprocess
    else:
        image_backbone = ResNetWithHooks(weights="IMAGENET1K_V2", hook_layer="layer4", freeze_backbone=True).to(device).eval()
        image_preprocess = None

    image_sae = ImageSAE(
        input_dim=model_config["img_input_dim"],
        hidden_dim=model_config["img_hidden_dim"],
        k_sparse=model_config["img_k_sparse"],
        use_bias=model_config["use_bias"],
        activation=model_config["activation"],
    ).to(device).eval()

    logger.info("Loading text backbone and SAE...")
    text_transforms = TextTransforms(
        tokenizer_name="bert-base-uncased",
        max_length=model_config["max_length"],
        d_text=model_config["d_model"],
    )
    vocab_size = text_transforms.get_vocab_size()
    text_backbone = create_transformer_with_hooks(
        vocab_size=vocab_size,
        config={
            "d_model": model_config["d_model"],
            "n_heads": model_config["n_heads"],
            "n_layers": model_config["n_layers"],
            "max_length": model_config["max_length"],
            "dropout": 0.1,
            "freeze_backbone": True,
            "return_features": True,
        },
    ).to(device).eval()
    state = checkpoint["model_state_dict"]
    if cfg.text_backbone_weights:
        text_backbone_path = Path(cfg.text_backbone_weights)
        if not text_backbone_path.is_file():
            raise FileNotFoundError(
                f"Text backbone checkpoint not found: {text_backbone_path}"
            )
        logger.info("Loading text backbone from %s", text_backbone_path)
        text_backbone.load_pretrained(str(text_backbone_path), strict=True)
    else:
        logger.info("Loading text backbone from the joint SAE checkpoint")
        text_backbone.load_state_dict(state["text_backbone"], strict=True)

    text_sae = create_text_sae(
        config={
            "d_model": model_config["d_model"],
            "sae_hidden_dim": model_config["txt_hidden_dim"],
            "k_sparse": model_config["txt_k_sparse"],
            "use_bias": model_config["use_bias"],
            "activation": model_config["activation"],
        }
    ).to(device).eval()

    image_sae.load_state_dict(state["image_sae"], strict=True)
    text_sae.load_state_dict(state["text_sae"], strict=True)

    for model in [image_backbone, image_sae, text_backbone, text_sae]:
        model.eval()
        for param in model.parameters():
            param.requires_grad_(False)

    return LoadedModels(
        image_backbone=image_backbone,
        image_sae=image_sae,
        text_backbone=text_backbone,
        text_sae=text_sae,
        image_preprocess=image_preprocess,
        model_config=model_config,
    )
