"""Global coupling optimizer for alternating local/global alignment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from .common import (
    clipped_prob,
    compute_image_hubness_numpy,
    compute_image_hubness_torch,
    compute_text_hubness_numpy,
    compute_text_hubness_torch,
    generalized_kl_torch,
    inverse_softplus,
)
from .objectives import global_objective


@dataclass
class GlobalSolverConfig:
    steps: int
    lr: float
    fit_weight: float
    gw_weight: float
    marginal_weight_v: float
    marginal_weight_t: float
    unary_weight: float
    unary_warmup_steps: int
    init_temperature: float
    row_active_tau: float
    col_active_tau: float
    hub_eps: float
    grad_clip: float = 5.0
    eps: float = 1e-8


def initialize_global_coupling(
    mu_v: np.ndarray,
    mu_t: np.ndarray,
    B: np.ndarray,
    temperature: float,
    eps: float = 1e-8,
) -> np.ndarray:
    mu_v = clipped_prob(mu_v)
    mu_t = clipped_prob(mu_t)
    base = np.outer(mu_v, mu_t).astype(np.float64)
    logits = np.exp(-B.astype(np.float64) / max(temperature, eps))
    T0 = base * logits
    T0 = T0 / max(T0.sum(), eps)
    return T0.astype(np.float32)


def optimize_global_coupling(
    *,
    init_T: np.ndarray,
    C_v: np.ndarray,
    C_t: np.ndarray,
    B: np.ndarray,
    bar_pi: np.ndarray,
    mu_v: np.ndarray,
    mu_t: np.ndarray,
    nu_v: np.ndarray,
    nu_t: np.ndarray,
    psi_t: np.ndarray,
    psi_v: np.ndarray,
    text_hub_weight: float,
    image_hub_weight: float,
    device: torch.device,
    config: GlobalSolverConfig,
    global_step_index: int,
) -> Tuple[np.ndarray, List[Dict[str, float]], Dict[str, np.ndarray | float]]:
    unary_weight = 0.0
    if config.unary_warmup_steps > 0:
        unary_weight = config.unary_weight * max(1.0 - (global_step_index / float(config.unary_warmup_steps)), 0.0)
    text_hub_weight = max(float(text_hub_weight), 0.0)
    image_hub_weight = max(float(image_hub_weight), 0.0)

    C_v_t = torch.as_tensor(C_v, dtype=torch.float32, device=device)
    C_t_t = torch.as_tensor(C_t, dtype=torch.float32, device=device)
    B_t = torch.as_tensor(B, dtype=torch.float32, device=device)
    bar_pi_t = torch.as_tensor(np.clip(np.asarray(bar_pi, dtype=np.float32), 0.0, None), dtype=torch.float32, device=device)
    mu_v_t = torch.as_tensor(np.clip(np.asarray(mu_v, dtype=np.float32), 0.0, None), dtype=torch.float32, device=device)
    mu_t_t = torch.as_tensor(np.clip(np.asarray(mu_t, dtype=np.float32), 0.0, None), dtype=torch.float32, device=device)
    nu_v_t = torch.as_tensor(clipped_prob(np.asarray(nu_v, dtype=np.float32), eps=config.hub_eps), dtype=torch.float32, device=device)
    nu_t_t = torch.as_tensor(clipped_prob(np.asarray(nu_t, dtype=np.float32), eps=config.hub_eps), dtype=torch.float32, device=device)
    psi_t_t = torch.as_tensor(clipped_prob(np.asarray(psi_t, dtype=np.float32), eps=config.hub_eps), dtype=torch.float32, device=device)
    psi_v_t = torch.as_tensor(clipped_prob(np.asarray(psi_v, dtype=np.float32), eps=config.hub_eps), dtype=torch.float32, device=device)

    init_T = np.clip(np.asarray(init_T, dtype=np.float32), config.eps, None)
    param = torch.nn.Parameter(inverse_softplus(torch.as_tensor(init_T, dtype=torch.float32, device=device)))
    optimizer = torch.optim.Adam([param], lr=config.lr)

    history: List[Dict[str, float]] = []
    best_T = init_T.copy()
    best_loss = None
    stagnant = 0

    for step in range(config.steps):
        T = F.softplus(param)
        loss = global_objective(
            T,
            C_v_t,
            C_t_t,
            B_t,
            bar_pi_t,
            mu_v_t,
            mu_t_t,
            nu_v_t,
            nu_t_t,
            psi_t_t,
            psi_v_t,
            fit_weight=config.fit_weight,
            gw_weight=config.gw_weight,
            marginal_weight_v=config.marginal_weight_v,
            marginal_weight_t=config.marginal_weight_t,
            unary_weight=unary_weight,
            text_hub_weight=text_hub_weight,
            image_hub_weight=image_hub_weight,
            row_active_tau=config.row_active_tau,
            col_active_tau=config.col_active_tau,
            eps=config.hub_eps,
        )
        optimizer.zero_grad()
        loss.backward()
        if config.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_([param], max_norm=config.grad_clip)
        optimizer.step()

        with torch.no_grad():
            T_current = F.softplus(param)
            loss_current = global_objective(
                T_current,
                C_v_t,
                C_t_t,
                B_t,
                bar_pi_t,
                mu_v_t,
                mu_t_t,
                nu_v_t,
                nu_t_t,
                psi_t_t,
                psi_v_t,
                fit_weight=config.fit_weight,
                gw_weight=config.gw_weight,
                marginal_weight_v=config.marginal_weight_v,
                marginal_weight_t=config.marginal_weight_t,
                unary_weight=unary_weight,
                text_hub_weight=text_hub_weight,
                image_hub_weight=image_hub_weight,
                row_active_tau=config.row_active_tau,
                col_active_tau=config.col_active_tau,
                eps=config.hub_eps,
            )
            text_hub_distribution = compute_text_hubness_torch(
                T_current,
                nu_v_t,
                row_active_tau=config.row_active_tau,
                eps=config.hub_eps,
            )
            image_hub_distribution = compute_image_hubness_torch(
                T_current,
                nu_t_t,
                col_active_tau=config.col_active_tau,
                eps=config.hub_eps,
            )
            text_hub_loss = generalized_kl_torch(text_hub_distribution, psi_t_t, eps=config.hub_eps)
            image_hub_loss = generalized_kl_torch(image_hub_distribution, psi_v_t, eps=config.hub_eps)

        loss_value = float(loss_current.detach().cpu())
        if best_loss is None or loss_value < best_loss:
            best_loss = loss_value
            best_T = T_current.detach().cpu().numpy().astype(np.float32)
            stagnant = 0
        else:
            stagnant += 1
            if stagnant >= 10:
                break

        if step % 10 == 0 or step == config.steps - 1:
            row_mass = T_current.sum(dim=1)
            col_mass = T_current.sum(dim=0)
            text_hub_ratio = (text_hub_distribution + config.hub_eps) / (psi_t_t + config.hub_eps)
            image_hub_ratio = (image_hub_distribution + config.hub_eps) / (psi_v_t + config.hub_eps)
            history.append(
                {
                    "step": int(step),
                    "global_step_index": int(global_step_index),
                    "loss": loss_value,
                    "unary_weight": float(unary_weight),
                    "text_hub_weight": float(text_hub_weight),
                    "image_hub_weight": float(image_hub_weight),
                    "hub_weight": float(text_hub_weight),
                    "text_hub_loss": float(text_hub_loss.detach().cpu()),
                    "image_hub_loss": float(image_hub_loss.detach().cpu()),
                    "hub_loss": float(text_hub_loss.detach().cpu()),
                    "text_hub_ratio_min": float(torch.min(text_hub_ratio).detach().cpu()),
                    "text_hub_ratio_max": float(torch.max(text_hub_ratio).detach().cpu()),
                    "image_hub_ratio_min": float(torch.min(image_hub_ratio).detach().cpu()),
                    "image_hub_ratio_max": float(torch.max(image_hub_ratio).detach().cpu()),
                    "hub_ratio_min": float(torch.min(text_hub_ratio).detach().cpu()),
                    "hub_ratio_max": float(torch.max(text_hub_ratio).detach().cpu()),
                    "mass": float(T_current.sum().detach().cpu()),
                    "row_entropy": float((-torch.sum(row_mass * torch.log(row_mass + config.eps))).detach().cpu()),
                    "col_entropy": float((-torch.sum(col_mass * torch.log(col_mass + config.eps))).detach().cpu()),
                    "text_hub_entropy": float((-torch.sum(text_hub_distribution * torch.log(text_hub_distribution + config.hub_eps))).detach().cpu()),
                    "image_hub_entropy": float((-torch.sum(image_hub_distribution * torch.log(image_hub_distribution + config.hub_eps))).detach().cpu()),
                    "hub_entropy": float((-torch.sum(text_hub_distribution * torch.log(text_hub_distribution + config.hub_eps))).detach().cpu()),
                }
            )

    best_text_hub_distribution = compute_text_hubness_numpy(
        best_T,
        nu_v=np.asarray(nu_v, dtype=np.float32),
        row_active_tau=config.row_active_tau,
        eps=config.hub_eps,
    )
    best_image_hub_distribution = compute_image_hubness_numpy(
        best_T,
        nu_t=np.asarray(nu_t, dtype=np.float32),
        col_active_tau=config.col_active_tau,
        eps=config.hub_eps,
    )
    psi_t_np = clipped_prob(np.asarray(psi_t, dtype=np.float32), eps=config.hub_eps)
    psi_v_np = clipped_prob(np.asarray(psi_v, dtype=np.float32), eps=config.hub_eps)
    best_text_hub_ratio = ((best_text_hub_distribution + config.hub_eps) / (psi_t_np + config.hub_eps)).astype(np.float32)
    best_image_hub_ratio = ((best_image_hub_distribution + config.hub_eps) / (psi_v_np + config.hub_eps)).astype(np.float32)
    diagnostics: Dict[str, np.ndarray | float] = {
        "text_hub_distribution": best_text_hub_distribution.astype(np.float32),
        "text_hub_ratio": best_text_hub_ratio,
        "text_hub_loss": float(
            np.sum(
                best_text_hub_distribution
                * (np.log(best_text_hub_distribution + config.hub_eps) - np.log(psi_t_np + config.hub_eps))
                - best_text_hub_distribution
                + psi_t_np
            )
        ),
        "image_hub_distribution": best_image_hub_distribution.astype(np.float32),
        "image_hub_ratio": best_image_hub_ratio,
        "image_hub_loss": float(
            np.sum(
                best_image_hub_distribution
                * (np.log(best_image_hub_distribution + config.hub_eps) - np.log(psi_v_np + config.hub_eps))
                - best_image_hub_distribution
                + psi_v_np
            )
        ),
        "text_hub_weight": float(text_hub_weight),
        "image_hub_weight": float(image_hub_weight),
        "hub_distribution": best_text_hub_distribution.astype(np.float32),
        "hub_ratio": best_text_hub_ratio,
        "hub_loss": float(
            np.sum(
                best_text_hub_distribution
                * (np.log(best_text_hub_distribution + config.hub_eps) - np.log(psi_t_np + config.hub_eps))
                - best_text_hub_distribution
                + psi_t_np
            )
        ),
        "hub_weight": float(text_hub_weight),
    }
    return best_T, history, diagnostics
