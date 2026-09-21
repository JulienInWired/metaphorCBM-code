"""Sample-level local alignment solver."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F

from .common import inverse_softplus
from .objectives import local_objective


@dataclass
class LocalSolverConfig:
    steps: int
    lr: float
    gw_weight: float
    unary_weight: float
    marginal_weight: float
    prior_weight: float
    prior_smoothing: float
    grad_clip: float = 5.0
    eps: float = 1e-8



def build_local_prior(global_block: np.ndarray, p_sum: float, q_sum: float, smoothing: float) -> np.ndarray:
    mass = 0.5 * (float(p_sum) + float(q_sum))
    prior = np.asarray(global_block, dtype=np.float32)
    prior = prior + float(smoothing)
    prior_sum = float(prior.sum())
    if prior_sum <= 0.0:
        prior = np.full_like(prior, fill_value=1.0 / max(prior.size, 1), dtype=np.float32)
        prior_sum = float(prior.sum())
    prior = prior / prior_sum
    return (prior * mass).astype(np.float32)



def solve_local_alignment(
    *,
    C_v: np.ndarray,
    C_t: np.ndarray,
    B: np.ndarray,
    p: np.ndarray,
    q: np.ndarray,
    prior_block: np.ndarray,
    device: torch.device,
    config: LocalSolverConfig,
) -> Optional[np.ndarray]:
    if p.size == 0 or q.size == 0:
        return None
    if float(p.sum()) <= config.eps or float(q.sum()) <= config.eps:
        return None

    prior = build_local_prior(prior_block, float(p.sum()), float(q.sum()), config.prior_smoothing)
    C_v_t = torch.as_tensor(C_v, dtype=torch.float32, device=device)
    C_t_t = torch.as_tensor(C_t, dtype=torch.float32, device=device)
    B_t = torch.as_tensor(B, dtype=torch.float32, device=device)
    p_t = torch.as_tensor(p, dtype=torch.float32, device=device)
    q_t = torch.as_tensor(q, dtype=torch.float32, device=device)
    prior_t = torch.as_tensor(prior, dtype=torch.float32, device=device)

    param = torch.nn.Parameter(inverse_softplus(torch.clamp(prior_t, min=config.eps)))
    optimizer = torch.optim.Adam([param], lr=config.lr)

    best_loss = None
    best_pi = None
    stagnant = 0

    for _ in range(config.steps):
        pi = F.softplus(param) + config.eps
        loss = local_objective(
            pi,
            C_v_t,
            C_t_t,
            B_t,
            p_t,
            q_t,
            prior_t,
            gw_weight=config.gw_weight,
            unary_weight=config.unary_weight,
            marginal_weight=config.marginal_weight,
            prior_weight=config.prior_weight,
        )
        optimizer.zero_grad()
        loss.backward()
        if config.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_([param], max_norm=config.grad_clip)
        optimizer.step()

        loss_value = float(loss.detach().cpu())
        if best_loss is None or loss_value < best_loss:
            best_loss = loss_value
            best_pi = pi.detach().cpu().numpy().astype(np.float32)
            stagnant = 0
        else:
            stagnant += 1
            if stagnant >= 5:
                break

    return best_pi
