"""Sparsity masks, regularizers, and activation statistics."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional


class KSparseMask(nn.Module):
    """Mask activations using a top-k threshold with randomized tie selection."""
    
    def __init__(self, k: int, dim: int = -1):
        """
        Configure the activation threshold.
        
        Args:
            k: Rank used to select the threshold.
            dim: Dimension along which to select activations.
        """
        super().__init__()
        self.k = k
        self.dim = dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return masked activations with the same shape as the input."""
        if self.k <= 0:
            return torch.zeros_like(x)
        
        if self.k >= x.size(self.dim):
            return x
        
        topk_values, topk_indices = torch.topk(x, self.k, dim=self.dim, largest=True)
        
        # Use the smallest of the top-k values as the threshold.
        threshold = topk_values[..., -1].unsqueeze(self.dim)
        
        mask = (x >= threshold).float()
        
        num_equal_threshold = (x == threshold).sum(dim=self.dim, keepdim=True)
        
        # Keep one randomly selected threshold tie per vector.
        if (num_equal_threshold > 1).any():
            equal_mask = (x == threshold).float()
            random_mask = torch.rand_like(equal_mask)
            equal_mask = equal_mask * random_mask
            
            _, equal_indices = torch.topk(equal_mask, 1, dim=self.dim)
            final_equal_mask = torch.zeros_like(equal_mask)
            final_equal_mask.scatter_(self.dim, equal_indices, 1.0)
            
            mask = ((x > threshold).float() + 
                   (x == threshold).float() * final_equal_mask)
        
        # Preserve masked values and double their gradient contribution.
        y = x * mask
        return y + (y - y.detach())
    
    def extra_repr(self) -> str:
        return f'k={self.k}, dim={self.dim}'


class TopKActivation(nn.Module):
    """Apply ReLU followed by a k-sparse mask."""
    
    def __init__(self, k: int, dim: int = -1):
        super().__init__()
        self.relu = nn.ReLU()
        self.k_sparse = KSparseMask(k, dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(x)
        return self.k_sparse(x)


def sparsity_loss(activations: torch.Tensor, 
                  target_sparsity: float = 0.05,
                  loss_type: str = 'l1') -> torch.Tensor:
    """
    Compute a sparsity regularizer.
    
    Args:
        activations: Activation tensor.
        target_sparsity: Target mean activation for the KL penalty.
        loss_type: One of 'l1', 'kl', or 'hoyer'.
        
    Returns:
        Scalar sparsity penalty.
    """
    if loss_type == 'l1':
        return torch.mean(torch.abs(activations))
    
    elif loss_type == 'kl':
        rho = target_sparsity
        rho_hat = torch.mean(activations, dim=0)
        
        # Keep logarithms finite.
        rho_hat = torch.clamp(rho_hat, min=1e-8, max=1-1e-8)
        
        kl_div = rho * torch.log(rho / rho_hat) + (1 - rho) * torch.log((1 - rho) / (1 - rho_hat))
        return torch.sum(kl_div)
    
    elif loss_type == 'hoyer':
        # Hoyer = (sqrt(n) - ||x||_1 / ||x||_2) / (sqrt(n) - 1)
        batch_size, n = activations.shape[0], activations.shape[-1]
        l1_norm = torch.norm(activations, p=1, dim=-1)
        l2_norm = torch.norm(activations, p=2, dim=-1)
        
        hoyer = (np.sqrt(n) - l1_norm / (l2_norm + 1e-8)) / (np.sqrt(n) - 1)
        # Negate the Hoyer score so minimization favors sparsity.
        return -torch.mean(hoyer)
    
    else:
        raise ValueError(f"Unsupported loss type: {loss_type}")


def compute_sparsity_metrics(activations: torch.Tensor) -> dict:
    """Summarize density, magnitude, variance, Gini score, and L1/L2 ratio."""
    with torch.no_grad():
        nonzero_ratio = (activations != 0).float().mean().item()
        
        mean_activation = torch.mean(torch.abs(activations)).item()
        
        activation_var = torch.var(activations).item()
        
        # Compute the Gini score over absolute activations.
        abs_acts = torch.abs(activations).flatten()
        sorted_acts, _ = torch.sort(abs_acts)
        n = len(sorted_acts)
        cumsum = torch.cumsum(sorted_acts, dim=0)
        gini = 1 - 2 * torch.sum(cumsum) / (n * cumsum[-1] + 1e-8)
        
        l1_norm = torch.norm(activations, p=1, dim=-1).mean()
        l2_norm = torch.norm(activations, p=2, dim=-1).mean()
        l1_l2_ratio = (l1_norm / (l2_norm + 1e-8)).item()
        
        return {
            'nonzero_ratio': nonzero_ratio,
            'mean_activation': mean_activation,
            'activation_var': activation_var,
            'gini_coefficient': gini.item(),
            'l1_l2_ratio': l1_l2_ratio
        }


class AdaptiveKSparse(nn.Module):
    """Update k using a running estimate of the active fraction."""
    
    def __init__(self, 
                 initial_k: int,
                 target_sparsity: float = 0.05,
                 adaptation_rate: float = 0.01,
                 min_k: int = 1,
                 max_k: Optional[int] = None):
        super().__init__()
        self.k = initial_k
        self.target_sparsity = target_sparsity
        self.adaptation_rate = adaptation_rate
        self.min_k = min_k
        self.max_k = max_k
        
        self.register_buffer('running_sparsity', torch.tensor(target_sparsity))
        self.register_buffer('update_count', torch.tensor(0))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        k_sparse = KSparseMask(self.k, dim=-1)
        output = k_sparse(x)
        
        if self.training:
            current_sparsity = (output != 0).float().mean()
            self.running_sparsity = (0.9 * self.running_sparsity + 
                                   0.1 * current_sparsity)
            self.update_count += 1
            
            # Update k every 100 training calls.
            if self.update_count % 100 == 0:
                self._adapt_k()
        
        return output
    
    def _adapt_k(self):
        """Adjust k when the active fraction differs from the target."""
        sparsity_error = self.running_sparsity - self.target_sparsity
        
        if abs(sparsity_error) > 0.01:
            if sparsity_error > 0:
                self.k = min(self.k + 1, self.max_k or float('inf'))
            else:
                self.k = max(self.k - 1, self.min_k)


def straight_through_round(x: torch.Tensor) -> torch.Tensor:
    """Round values in the forward pass and pass gradients through unchanged."""
    return x.round() + (x - x.detach())


class GumbelTopK(nn.Module):
    """Select activations with Gumbel noise and a differentiable top-k mask."""
    
    def __init__(self, k: int, temperature: float = 1.0, hard: bool = True):
        super().__init__()
        self.k = k
        self.temperature = temperature
        self.hard = hard
    
    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits) + 1e-8) + 1e-8)
        noisy_logits = (logits + gumbel_noise) / self.temperature
        
        # Center the soft mask on the top-k threshold.
        topk_values, topk_indices = torch.topk(noisy_logits, self.k, dim=-1)
        threshold = topk_values[..., -1:].expand_as(logits)
        
        soft_mask = torch.sigmoid((noisy_logits - threshold) / self.temperature)
        
        if self.hard:
            # Use hard selections in the forward pass and soft-mask gradients.
            hard_mask = (noisy_logits >= threshold).float()
            return hard_mask + (soft_mask - soft_mask.detach())
        else:
            return soft_mask 
