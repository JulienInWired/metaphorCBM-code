"""Objective functions for local and global coupling optimization."""

from __future__ import annotations

import torch

from .common import compute_image_hubness_torch, compute_text_hubness_torch, generalized_kl_torch


def gw_loss_dense(C_v: torch.Tensor, C_t: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
    """Canonical GW loss for square loss with a possibly unbalanced coupling."""
    r = T.sum(dim=1)
    c = T.sum(dim=0)
    term_v = torch.dot((C_v.square() @ r), r)
    term_t = torch.dot((C_t.square() @ c), c)
    cross = torch.sum((C_v @ T @ C_t) * T)
    return term_v + term_t - 2.0 * cross


def local_objective(
    pi: torch.Tensor,
    C_v: torch.Tensor,
    C_t: torch.Tensor,
    B: torch.Tensor,
    p: torch.Tensor,
    q: torch.Tensor,
    prior: torch.Tensor,
    *,
    gw_weight: float,
    unary_weight: float,
    marginal_weight: float,
    prior_weight: float,
) -> torch.Tensor:
    return (
        gw_weight * gw_loss_dense(C_v, C_t, pi)
        + unary_weight * torch.sum(B * pi)
        + marginal_weight * (generalized_kl_torch(pi.sum(dim=1), p) + generalized_kl_torch(pi.sum(dim=0), q))
        + prior_weight * generalized_kl_torch(pi, prior)
    )


def text_hubness_objective(
    T: torch.Tensor,
    nu_v: torch.Tensor,
    psi_t: torch.Tensor,
    *,
    row_active_tau: float,
    eps: float = 1e-8,
) -> torch.Tensor:
    hubness = compute_text_hubness_torch(T, nu_v, row_active_tau=row_active_tau, eps=eps)
    return generalized_kl_torch(hubness, psi_t, eps=eps)


def image_hubness_objective(
    T: torch.Tensor,
    nu_t: torch.Tensor,
    psi_v: torch.Tensor,
    *,
    col_active_tau: float,
    eps: float = 1e-8,
) -> torch.Tensor:
    hubness = compute_image_hubness_torch(T, nu_t, col_active_tau=col_active_tau, eps=eps)
    return generalized_kl_torch(hubness, psi_v, eps=eps)


def global_objective(
    T: torch.Tensor,
    C_v: torch.Tensor,
    C_t: torch.Tensor,
    B: torch.Tensor,
    bar_pi: torch.Tensor,
    mu_v: torch.Tensor,
    mu_t: torch.Tensor,
    nu_v: torch.Tensor,
    nu_t: torch.Tensor,
    psi_t: torch.Tensor,
    psi_v: torch.Tensor,
    *,
    fit_weight: float,
    gw_weight: float,
    marginal_weight_v: float,
    marginal_weight_t: float,
    unary_weight: float,
    text_hub_weight: float,
    image_hub_weight: float,
    row_active_tau: float,
    col_active_tau: float,
    eps: float = 1e-8,
) -> torch.Tensor:
    return (
        fit_weight * generalized_kl_torch(bar_pi, T, eps=eps)
        + gw_weight * gw_loss_dense(C_v, C_t, T)
        + marginal_weight_v * generalized_kl_torch(T.sum(dim=1), mu_v, eps=eps)
        + marginal_weight_t * generalized_kl_torch(T.sum(dim=0), mu_t, eps=eps)
        + unary_weight * torch.sum(B * T)
        + text_hub_weight * text_hubness_objective(T, nu_v, psi_t, row_active_tau=row_active_tau, eps=eps)
        + image_hub_weight * image_hubness_objective(T, nu_t, psi_v, col_active_tau=col_active_tau, eps=eps)
    )
