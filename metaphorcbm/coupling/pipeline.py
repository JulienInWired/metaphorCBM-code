"""Main alternating-optimization pipeline for cross-modal coupling."""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import torch
from tqdm import tqdm

from metaphorcbm.data import ImageTransforms, TextTransforms

from .common import (
    clipped_prob,
    compute_family_compressed_target,
    compute_hubness_gains,
    compute_hubness_target,
    compute_image_hubness_numpy,
    compute_text_hubness_numpy,
    configure_logger,
    ensure_dir,
    linear_warmup_factor,
    select_topk_indices,
    set_seed,
    update_hubness_ema,
)
from .concept_basis import ConceptBasis, ReducedSampleRecord, build_concept_basis
from .config import CouplingRunConfig
from .datasets import build_dataloader
from .export import expand_reduced_coupling, save_artifacts
from .feature_extraction import ExtractionResult, extract_sparse_activations
from .global_solver import GlobalSolverConfig, initialize_global_coupling, optimize_global_coupling
from .local_solver import LocalSolverConfig, solve_local_alignment
from .model_loader import load_models


class CrossModalCouplingTrainer:
    def __init__(self, config: CouplingRunConfig):
        self.config = config
        self.output_dir = ensure_dir(config.output)
        self.logger = configure_logger(self.output_dir, name="coupling")
        set_seed(config.seed)
        self.device = torch.device(config.device)

    def _build_dataloader(self, models) -> torch.utils.data.DataLoader:
        if getattr(models, "image_preprocess", None) is not None:
            image_transforms = models.image_preprocess
        else:
            image_transforms = ImageTransforms(resize_size=256, crop_size=224, imagenet_normalize=True)
        text_transforms = TextTransforms(
            tokenizer_name="bert-base-uncased",
            max_length=models.model_config["max_length"],
            d_text=models.model_config["d_model"],
        )
        dataloader = build_dataloader(
            root_dir=self.config.data_root,
            ann_file=self.config.ann_file,
            image_transforms=image_transforms,
            text_transforms=text_transforms,
            split=self.config.split,
            max_samples=self.config.max_samples,
            seed=self.config.seed,
            balanced_sampling=self.config.balanced_sampling,
            caption_mode=self.config.caption_mode,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
        )
        self.logger.info("Coupling dataloader built: %d batches, caption_mode=%s", len(dataloader), self.config.caption_mode)
        return dataloader

    def _extract_activations(self, models) -> ExtractionResult:
        dataloader = self._build_dataloader(models)
        return extract_sparse_activations(dataloader, models, self.device, self.logger)

    def _build_basis(self, extraction: ExtractionResult, models) -> ConceptBasis:
        img_decoder = models.image_sae.decoder[-1].weight
        txt_decoder = models.text_sae.decoder[-1].weight
        basis = build_concept_basis(
            records=extraction.records,
            img_matrix=extraction.img_matrix,
            txt_matrix=extraction.txt_matrix,
            img_support_counts=extraction.img_support_counts,
            img_peak_support_counts=extraction.img_peak_support_counts,
            img_peak_dominant_patch=extraction.img_peak_dominant_patch,
            img_peak_dominant_mass=extraction.img_peak_dominant_mass,
            txt_support_counts=extraction.txt_support_counts,
            txt_peak_support_counts=extraction.txt_peak_support_counts,
            txt_peak_consistency=extraction.txt_peak_consistency,
            txt_peak_dominant_word=extraction.txt_peak_dominant_word,
            txt_peak_dominant_mass=extraction.txt_peak_dominant_mass,
            txt_peak_stopword_mass=extraction.txt_peak_stopword_mass,
            txt_peak_invalid_mass=extraction.txt_peak_invalid_mass,
            img_decoder_weights=img_decoder,
            txt_decoder_weights=txt_decoder,
            activation_power=self.config.activation_power,
            reliability_eps=self.config.reliability_eps,
            image_prefilter_enabled=self.config.image_prefilter_enabled,
            image_prefilter_min_peak_support=self.config.image_prefilter_min_peak_support,
            image_prefilter_dom_patch_threshold=self.config.image_prefilter_dom_patch_threshold,
            text_prefilter_enabled=self.config.text_prefilter_enabled,
            text_prefilter_remove_stopwords=self.config.text_prefilter_remove_stopwords,
            text_prefilter_remove_inconsistent=self.config.text_prefilter_remove_inconsistent,
            text_prefilter_stop_min_support=self.config.text_prefilter_stop_min_support,
            text_prefilter_stop_dom_threshold=self.config.text_prefilter_stop_dom_threshold,
            text_prefilter_stop_mass_threshold=self.config.text_prefilter_stop_mass_threshold,
            text_prefilter_consistency_min_support=self.config.text_prefilter_consistency_min_support,
            text_prefilter_p_threshold=self.config.text_prefilter_p_threshold,
            text_prefilter_dom_content_threshold=self.config.text_prefilter_dom_content_threshold,
        )
        num_img_total = int(len(basis.image_prefilter.keep_mask))
        num_img_kept = int(basis.image_prefilter.keep_mask.sum())
        num_txt_total = int(len(basis.text_prefilter.keep_mask))
        num_txt_kept = int(basis.text_prefilter.keep_mask.sum())
        self.logger.info(
            "Full concept basis ready: image=%d/%d kept after prefiltering, text=%d/%d kept after prefiltering",
            num_img_kept,
            num_img_total,
            num_txt_kept,
            num_txt_total,
        )
        img_removed_reasons, img_removed_counts = np.unique(
            basis.image_prefilter.filter_reason[~basis.image_prefilter.keep_mask],
            return_counts=True,
        )
        if img_removed_counts.size > 0:
            reason_summary = ", ".join(
                f"{str(reason)}={int(count)}" for reason, count in zip(img_removed_reasons.tolist(), img_removed_counts.tolist())
            )
            self.logger.info("Image concept prefilter removed: %s", reason_summary)
        removed_reasons, removed_counts = np.unique(
            basis.text_prefilter.filter_reason[~basis.text_prefilter.keep_mask],
            return_counts=True,
        )
        if removed_counts.size > 0:
            reason_summary = ", ".join(
                f"{str(reason)}={int(count)}" for reason, count in zip(removed_reasons.tolist(), removed_counts.tolist())
            )
            self.logger.info("Text concept prefilter removed: %s", reason_summary)
        return basis

    def _compute_hub_feedback_state(
        self,
        *,
        T: np.ndarray,
        nu_v: np.ndarray,
        nu_t: np.ndarray,
        psi_t: np.ndarray,
        psi_v: np.ndarray,
        global_step_index: int,
        previous_text_ema: np.ndarray | None,
        previous_image_ema: np.ndarray | None,
    ) -> Dict[str, np.ndarray | float]:
        warmup_factor = linear_warmup_factor(
            global_step_index,
            self.config.hub_warmup_start_steps,
            self.config.hub_warmup_ramp_steps,
        )
        hub_raw = compute_text_hubness_numpy(
            T,
            nu_v=nu_v,
            row_active_tau=self.config.hub_row_active_tau,
            eps=self.config.hub_eps,
        )
        image_hub_raw = compute_image_hubness_numpy(
            T,
            nu_t=nu_t,
            col_active_tau=self.config.hub_row_active_tau,
            eps=self.config.hub_eps,
        )
        text_hub_ema = update_hubness_ema(previous_text_ema, hub_raw, decay=self.config.hub_ema_decay)
        image_hub_ema = update_hubness_ema(previous_image_ema, image_hub_raw, decay=self.config.hub_ema_decay)
        text_gains, text_ratio = compute_hubness_gains(
            text_hub_ema,
            psi_t,
            gamma=self.config.hub_feedback_gamma * warmup_factor,
            gain_min=self.config.hub_gain_min,
            gain_max=self.config.hub_gain_max,
            eps=self.config.hub_eps,
        )
        image_gains, image_ratio = compute_hubness_gains(
            image_hub_ema,
            psi_v,
            gamma=self.config.hub_feedback_gamma * warmup_factor,
            gain_min=self.config.hub_gain_min,
            gain_max=self.config.hub_gain_max,
            eps=self.config.hub_eps,
        )
        return {
            "warmup_factor": float(warmup_factor),
            "text_hub_raw": hub_raw.astype(np.float32),
            "text_hub_ema": text_hub_ema.astype(np.float32),
            "text_gains": text_gains.astype(np.float32),
            "text_ratio": text_ratio.astype(np.float32),
            "image_hub_raw": image_hub_raw.astype(np.float32),
            "image_hub_ema": image_hub_ema.astype(np.float32),
            "image_gains": image_gains.astype(np.float32),
            "image_ratio": image_ratio.astype(np.float32),
            "hub_raw": hub_raw.astype(np.float32),
            "hub_ema": text_hub_ema.astype(np.float32),
            "gains": text_gains.astype(np.float32),
            "ratio": text_ratio.astype(np.float32),
        }

    def _local_batch(
        self,
        records: List[ReducedSampleRecord],
        basis: ConceptBasis,
        T: np.ndarray,
        local_cfg: LocalSolverConfig,
        image_hub_gains: np.ndarray | None,
        text_hub_gains: np.ndarray | None,
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        n_img, n_txt = T.shape
        aggregate = np.zeros((n_img, n_txt), dtype=np.float32)
        total_samples = len(records)
        valid_count = 0
        skipped_empty = 0
        skipped_solver = 0
        local_masses: List[float] = []

        for record in records:
            img_top = self._select_local_image_subset(record)
            txt_top = self._select_local_text_subset(record)
            if img_top is None or txt_top is None:
                skipped_empty += 1
                continue
            img_subset_idx, img_subset_mass = img_top
            txt_subset_idx, txt_subset_mass = txt_top

            prior_block = T[np.ix_(img_subset_idx, txt_subset_idx)].astype(np.float32, copy=False)
            if image_hub_gains is not None and img_subset_idx.size > 0:
                prior_block = np.asarray(image_hub_gains[img_subset_idx], dtype=np.float32)[:, None] * prior_block
            if text_hub_gains is not None and txt_subset_idx.size > 0:
                prior_block = prior_block * np.asarray(text_hub_gains[txt_subset_idx], dtype=np.float32)[None, :]

            local_pi = solve_local_alignment(
                C_v=basis.C_v[np.ix_(img_subset_idx, img_subset_idx)],
                C_t=basis.C_t[np.ix_(txt_subset_idx, txt_subset_idx)],
                B=basis.B[np.ix_(img_subset_idx, txt_subset_idx)],
                p=img_subset_mass,
                q=txt_subset_mass,
                prior_block=prior_block,
                device=self.device,
                config=local_cfg,
            )
            if local_pi is None or not np.isfinite(local_pi).all():
                skipped_solver += 1
                continue
            mass = float(local_pi.sum())
            if mass <= 0.0:
                skipped_solver += 1
                continue
            aggregate[np.ix_(img_subset_idx, txt_subset_idx)] += (local_pi / mass).astype(np.float32)
            valid_count += 1
            local_masses.append(mass)

        if total_samples > 0:
            aggregate /= float(total_samples)
        else:
            aggregate[:] = 0.0

        stats = {
            "total_samples": float(total_samples),
            "valid_samples": float(valid_count),
            "skipped_empty": float(skipped_empty),
            "skipped_solver": float(skipped_solver),
            "mean_local_mass": float(np.mean(local_masses) if local_masses else 0.0),
        }
        return aggregate, stats

    def _iter_record_batches(self, records: List[ReducedSampleRecord]):
        # Keep a deterministic traversal order across epochs for reproducibility.
        batch_size = max(1, int(self.config.optimization_batch_size))
        for start in range(0, len(records), batch_size):
            yield records[start : start + batch_size]

    def _select_local_image_subset(self, record: ReducedSampleRecord):
        if record.img_idx.size == 0:
            return None
        # Select visual support by raw activation strength.
        top_positions = select_topk_indices(record.img_values, self.config.local_topk)
        subset_idx = record.img_idx[top_positions]
        subset_mass = record.img_mass[top_positions].astype(np.float32)
        return subset_idx, subset_mass

    def _select_local_text_subset(self, record: ReducedSampleRecord):
        if record.txt_idx.size == 0:
            return None

        # Rank text concepts by activation mass on the prefiltered basis.
        top_positions = select_topk_indices(record.txt_mass, self.config.local_topk)
        subset_idx = record.txt_idx[top_positions]
        subset_mass = record.txt_mass[top_positions].astype(np.float32)
        return subset_idx, subset_mass

    def run(self) -> Dict[str, str]:
        self.logger.info("Starting coupling pipeline...")
        models = load_models(self.config, self.device, self.logger)
        extraction = self._extract_activations(models)
        basis = self._build_basis(extraction, models)

        if basis.C_v.shape[0] == 0 or basis.C_t.shape[0] == 0:
            raise RuntimeError("Loaded SAE hidden dimensions are zero; coupling cannot be trained.")

        local_cfg = LocalSolverConfig(
            steps=self.config.local_steps,
            lr=self.config.local_lr,
            gw_weight=self.config.local_gw_weight,
            unary_weight=self.config.local_unary_weight,
            marginal_weight=self.config.local_marginal_weight,
            prior_weight=self.config.local_prior_weight,
            prior_smoothing=self.config.local_prior_smoothing,
        )
        global_cfg = GlobalSolverConfig(
            steps=self.config.global_steps,
            lr=self.config.global_lr,
            fit_weight=self.config.global_fit_weight,
            gw_weight=self.config.global_gw_weight,
            marginal_weight_v=self.config.global_marginal_weight_v,
            marginal_weight_t=self.config.global_marginal_weight_t,
            unary_weight=self.config.global_unary_weight,
            unary_warmup_steps=0,
            init_temperature=self.config.init_temperature,
            row_active_tau=self.config.hub_row_active_tau,
            col_active_tau=self.config.hub_row_active_tau,
            hub_eps=self.config.hub_eps,
        )

        T = initialize_global_coupling(basis.mu_v, basis.mu_t, basis.B, temperature=self.config.init_temperature)
        initial_T = T.copy()
        round_trace: List[Dict] = []
        global_update_trace: List[Dict] = []
        num_samples = len(basis.reduced_records)
        optimization_batch_size = max(1, int(self.config.optimization_batch_size))
        num_batches = (num_samples + optimization_batch_size - 1) // optimization_batch_size if num_samples > 0 else 0
        if self.config.global_unary_warmup_steps is not None:
            global_cfg.unary_warmup_steps = max(0, int(self.config.global_unary_warmup_steps))
        else:
            global_cfg.unary_warmup_steps = max(0, int(self.config.global_unary_warmup_rounds)) * max(num_batches, 1)

        nu_v = clipped_prob(basis.mu_v, eps=self.config.hub_eps)
        nu_t = clipped_prob(basis.mu_t, eps=self.config.hub_eps)
        psi_v = compute_hubness_target(basis.mu_v, power=self.config.hub_target_power, eps=self.config.hub_eps)
        psi_t_base = compute_hubness_target(basis.mu_t, power=self.config.hub_target_power, eps=self.config.hub_eps)
        psi_t = psi_t_base
        family_target_info: Dict[str, np.ndarray] | None = None
        if self.config.family_target_enabled:
            psi_t, family_target_info = compute_family_compressed_target(
                psi_t_base,
                basis.selection.txt_peak_dominant_word,
                compression=self.config.family_target_compression,
                eps=self.config.hub_eps,
            )
            self.logger.info(
                "Applied duplicate-compressed text family target: families=%d compression=%.4f",
                int(np.asarray(family_target_info["family_names"]).size),
                float(self.config.family_target_compression),
            )

        global_update_step = 0
        hub_state = self._compute_hub_feedback_state(
            T=T,
            nu_v=nu_v,
            nu_t=nu_t,
            psi_t=psi_t,
            psi_v=psi_v,
            global_step_index=global_update_step,
            previous_text_ema=None,
            previous_image_ema=None,
        )
        previous_round_T = T.copy()
        latest_global_diagnostics: Dict[str, np.ndarray | float] = {
            "text_hub_distribution": hub_state["text_hub_raw"],
            "text_hub_ratio": hub_state["text_ratio"],
            "text_hub_loss": float(
                np.sum(
                    np.asarray(hub_state["text_hub_raw"], dtype=np.float32)
                    * (
                        np.log(np.asarray(hub_state["text_hub_raw"], dtype=np.float32) + self.config.hub_eps)
                        - np.log(psi_t + self.config.hub_eps)
                    )
                    - np.asarray(hub_state["text_hub_raw"], dtype=np.float32)
                    + psi_t
                )
            ),
            "image_hub_distribution": hub_state["image_hub_raw"],
            "image_hub_ratio": hub_state["image_ratio"],
            "image_hub_loss": float(
                np.sum(
                    np.asarray(hub_state["image_hub_raw"], dtype=np.float32)
                    * (
                        np.log(np.asarray(hub_state["image_hub_raw"], dtype=np.float32) + self.config.hub_eps)
                        - np.log(psi_v + self.config.hub_eps)
                    )
                    - np.asarray(hub_state["image_hub_raw"], dtype=np.float32)
                    + psi_v
                )
            ),
            "text_hub_weight": 0.0,
            "image_hub_weight": 0.0,
            "hub_distribution": hub_state["text_hub_raw"],
            "hub_ratio": hub_state["text_ratio"],
            "hub_loss": 0.0,
            "hub_weight": 0.0,
        }
        latest_global_diagnostics["hub_loss"] = float(latest_global_diagnostics["text_hub_loss"])
        self.logger.info(
            "Optimization schedule: extraction_batch_size=%d optimization_batch_size=%d global_steps_per_batch=%d unary_warmup_steps=%d hub_warmup=(start=%d,ramp=%d)",
            int(self.config.batch_size),
            optimization_batch_size,
            global_cfg.steps,
            global_cfg.unary_warmup_steps,
            int(self.config.hub_warmup_start_steps),
            int(self.config.hub_warmup_ramp_steps),
        )

        for round_idx in range(self.config.outer_iterations):
            self.logger.info(
                "Alternating round %d/%d with %d optimization batches",
                round_idx + 1,
                self.config.outer_iterations,
                num_batches,
            )
            round_valid = 0.0
            round_skipped_empty = 0.0
            round_skipped_solver = 0.0
            round_local_mass_sum = 0.0
            round_nonempty_batches = 0
            round_batch_trace: List[Dict[str, float]] = []

            batch_iterator = tqdm(
                enumerate(self._iter_record_batches(basis.reduced_records), start=1),
                total=num_batches,
                desc=f"Alternating round {round_idx + 1}",
                leave=False,
            )
            for batch_idx, batch_records in batch_iterator:
                bar_pi, local_stats = self._local_batch(
                    batch_records,
                    basis,
                    T,
                    local_cfg,
                    image_hub_gains=np.asarray(hub_state["image_gains"], dtype=np.float32),
                    text_hub_gains=np.asarray(hub_state["text_gains"], dtype=np.float32),
                )
                effective_text_hub_weight = float(self.config.global_hub_weight) * float(hub_state["warmup_factor"])
                effective_image_hub_weight = float(self.config.global_image_hub_weight) * float(hub_state["warmup_factor"])
                T, global_history, latest_global_diagnostics = optimize_global_coupling(
                    init_T=T,
                    C_v=basis.C_v,
                    C_t=basis.C_t,
                    B=basis.B,
                    bar_pi=bar_pi,
                    mu_v=basis.mu_v,
                    mu_t=basis.mu_t,
                    nu_v=nu_v,
                    nu_t=nu_t,
                    psi_t=psi_t,
                    psi_v=psi_v,
                    text_hub_weight=effective_text_hub_weight,
                    image_hub_weight=effective_image_hub_weight,
                    device=self.device,
                    config=global_cfg,
                    global_step_index=global_update_step,
                )
                global_update_step += 1
                hub_state = self._compute_hub_feedback_state(
                    T=T,
                    nu_v=nu_v,
                    nu_t=nu_t,
                    psi_t=psi_t,
                    psi_v=psi_v,
                    global_step_index=global_update_step,
                    previous_text_ema=np.asarray(hub_state["text_hub_ema"], dtype=np.float32),
                    previous_image_ema=np.asarray(hub_state["image_hub_ema"], dtype=np.float32),
                )

                for entry in global_history:
                    entry_with_context = dict(entry)
                    entry_with_context["round"] = int(round_idx + 1)
                    entry_with_context["batch_index"] = int(batch_idx)
                    global_update_trace.append(entry_with_context)
                    round_batch_trace.append(entry_with_context)

                round_valid += local_stats["valid_samples"]
                round_skipped_empty += local_stats["skipped_empty"]
                round_skipped_solver += local_stats["skipped_solver"]
                if local_stats["total_samples"] > 0:
                    round_local_mass_sum += local_stats["mean_local_mass"]
                    round_nonempty_batches += 1
                if batch_idx == 1 or batch_idx == num_batches or batch_idx % 10 == 0:
                    current_text_hub_ratio = np.asarray(latest_global_diagnostics["text_hub_ratio"], dtype=np.float32)
                    current_image_hub_ratio = np.asarray(latest_global_diagnostics["image_hub_ratio"], dtype=np.float32)
                    self.logger.info(
                        "Round %d batch %d/%d: valid=%d skipped_empty=%d skipped_solver=%d bar_pi_sum=%.6f T_mass=%.6f text_hub_loss=%.6f text_hub_ratio_max=%.4f image_hub_loss=%.6f image_hub_ratio_max=%.4f",
                        round_idx + 1,
                        batch_idx,
                        num_batches,
                        int(local_stats["valid_samples"]),
                        int(local_stats["skipped_empty"]),
                        int(local_stats["skipped_solver"]),
                        float(bar_pi.sum()),
                        float(T.sum()),
                        float(latest_global_diagnostics["text_hub_loss"]),
                        float(current_text_hub_ratio.max()) if current_text_hub_ratio.size > 0 else 0.0,
                        float(latest_global_diagnostics["image_hub_loss"]),
                        float(current_image_hub_ratio.max()) if current_image_hub_ratio.size > 0 else 0.0,
                    )

            mean_local_mass = round_local_mass_sum / max(round_nonempty_batches, 1)
            round_t_mass = float(T.sum())
            prev_mass = float(previous_round_T.sum())
            delta_T = T - previous_round_T
            delta_T_l1 = float(np.abs(delta_T).sum())
            delta_T_mean_abs = float(np.mean(np.abs(delta_T)))
            prev_norm = float(np.linalg.norm(previous_round_T))
            delta_T_fro = float(np.linalg.norm(delta_T))
            delta_T_fro_rel = delta_T_fro / max(prev_norm, 1e-12)
            text_hub_ratio_final = np.asarray(latest_global_diagnostics["text_hub_ratio"], dtype=np.float32)
            image_hub_ratio_final = np.asarray(latest_global_diagnostics["image_hub_ratio"], dtype=np.float32)
            self.logger.info(
                "Round %d summary: valid=%d skipped_empty=%d skipped_solver=%d mean_batch_local_mass=%.4f T_mass=%.6f delta_T_fro_rel=%.6e delta_T_mean_abs=%.6e text_hub_ratio_max=%.4f image_hub_ratio_max=%.4f",
                round_idx + 1,
                int(round_valid),
                int(round_skipped_empty),
                int(round_skipped_solver),
                mean_local_mass,
                round_t_mass,
                delta_T_fro_rel,
                delta_T_mean_abs,
                float(text_hub_ratio_final.max()) if text_hub_ratio_final.size > 0 else 0.0,
                float(image_hub_ratio_final.max()) if image_hub_ratio_final.size > 0 else 0.0,
            )
            round_trace.append(
                {
                    "round": int(round_idx + 1),
                    "num_batches": int(num_batches),
                    "global_update_step": int(global_update_step),
                    "local_stats": {
                        "total_samples": float(num_samples),
                        "valid_samples": float(round_valid),
                        "skipped_empty": float(round_skipped_empty),
                        "skipped_solver": float(round_skipped_solver),
                        "mean_batch_local_mass": float(mean_local_mass),
                    },
                    "global_history": round_batch_trace,
                    "final_T_mass": round_t_mass,
                    "previous_round_T_mass": prev_mass,
                    "delta_T_l1": delta_T_l1,
                    "delta_T_mean_abs": delta_T_mean_abs,
                    "delta_T_fro": delta_T_fro,
                    "delta_T_fro_rel": delta_T_fro_rel,
                    "text_hub_ratio_min": float(text_hub_ratio_final.min()) if text_hub_ratio_final.size > 0 else 0.0,
                    "text_hub_ratio_max": float(text_hub_ratio_final.max()) if text_hub_ratio_final.size > 0 else 0.0,
                    "image_hub_ratio_min": float(image_hub_ratio_final.min()) if image_hub_ratio_final.size > 0 else 0.0,
                    "image_hub_ratio_max": float(image_hub_ratio_final.max()) if image_hub_ratio_final.size > 0 else 0.0,
                    "text_hub_loss": float(latest_global_diagnostics["text_hub_loss"]),
                    "image_hub_loss": float(latest_global_diagnostics["image_hub_loss"]),
                    "hub_ratio_min": float(text_hub_ratio_final.min()) if text_hub_ratio_final.size > 0 else 0.0,
                    "hub_ratio_max": float(text_hub_ratio_final.max()) if text_hub_ratio_final.size > 0 else 0.0,
                    "hub_loss": float(latest_global_diagnostics["text_hub_loss"]),
                }
            )
            previous_round_T = T.copy()

        full_shape = (models.model_config["img_hidden_dim"], models.model_config["txt_hidden_dim"])
        full_T = expand_reduced_coupling(
            reduced_T=T,
            img_original_indices=basis.selection.basis_img_indices,
            txt_original_indices=basis.selection.basis_txt_indices,
            full_shape=full_shape,
        )
        full_initial_T = expand_reduced_coupling(
            reduced_T=initial_T,
            img_original_indices=basis.selection.basis_img_indices,
            txt_original_indices=basis.selection.basis_txt_indices,
            full_shape=full_shape,
        )

        hub_diagnostics = {
            "nu_v": nu_v.astype(np.float32),
            "nu_t": nu_t.astype(np.float32),
            "psi_t_base": psi_t_base.astype(np.float32),
            "psi_t": psi_t.astype(np.float32),
            "psi_v": psi_v.astype(np.float32),
            "text_hub_raw_final": np.asarray(latest_global_diagnostics["text_hub_distribution"], dtype=np.float32),
            "text_hub_raw_ratio_final": np.asarray(latest_global_diagnostics["text_hub_ratio"], dtype=np.float32),
            "text_hub_ema_final": np.asarray(hub_state["text_hub_ema"], dtype=np.float32),
            "text_hub_gain_final": np.asarray(hub_state["text_gains"], dtype=np.float32),
            "text_hub_ratio_final": np.asarray(hub_state["text_ratio"], dtype=np.float32),
            "image_hub_raw_final": np.asarray(latest_global_diagnostics["image_hub_distribution"], dtype=np.float32),
            "image_hub_raw_ratio_final": np.asarray(latest_global_diagnostics["image_hub_ratio"], dtype=np.float32),
            "image_hub_ema_final": np.asarray(hub_state["image_hub_ema"], dtype=np.float32),
            "image_hub_gain_final": np.asarray(hub_state["image_gains"], dtype=np.float32),
            "image_hub_ratio_final": np.asarray(hub_state["image_ratio"], dtype=np.float32),
            "hub_raw_final": np.asarray(latest_global_diagnostics["text_hub_distribution"], dtype=np.float32),
            "hub_raw_ratio_final": np.asarray(latest_global_diagnostics["text_hub_ratio"], dtype=np.float32),
            "hub_ema_final": np.asarray(hub_state["text_hub_ema"], dtype=np.float32),
            "hub_gain_final": np.asarray(hub_state["text_gains"], dtype=np.float32),
            "hub_ratio_final": np.asarray(hub_state["text_ratio"], dtype=np.float32),
            "hub_feedback_warmup_final": float(hub_state["warmup_factor"]),
        }
        if family_target_info is not None:
            family_ids = np.asarray(family_target_info["family_ids"], dtype=np.int64)
            family_names = np.asarray(family_target_info["family_names"], dtype="<U128")
            family_target = np.asarray(family_target_info["family_target"], dtype=np.float32)
            text_hub = np.asarray(latest_global_diagnostics["text_hub_distribution"], dtype=np.float32)
            family_hub = np.bincount(family_ids, weights=text_hub.astype(np.float64), minlength=int(family_names.size))
            hub_diagnostics.update(
                {
                    "text_family_ids": family_ids,
                    "text_family_names": family_names,
                    "text_family_base_target": np.asarray(family_target_info["family_base_target"], dtype=np.float32),
                    "text_family_target": family_target,
                    "text_family_effective_size": np.asarray(family_target_info["family_effective_size"], dtype=np.float32),
                    "text_family_hub_final": family_hub.astype(np.float32),
                    "text_family_ratio_final": ((family_hub + self.config.hub_eps) / (family_target + self.config.hub_eps)).astype(np.float32),
                }
            )

        outputs = save_artifacts(
            output_dir=str(self.output_dir),
            config=self.config.to_dict(),
            full_coupling_dense=full_T,
            initial_coupling_dense=full_initial_T,
            concept_basis=basis,
            sample_records=extraction.records,
            optimization_trace={"rounds": round_trace, "global_updates": global_update_trace},
            model_config=models.model_config,
            hub_diagnostics=hub_diagnostics,
        )
        self.logger.info("Coupling pipeline completed. Outputs saved to %s", self.output_dir)
        return outputs
