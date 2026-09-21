"""Reconstruction, sparsity, and cross-modal loss components for SAE training."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, Union


def reconstruction_loss(original: torch.Tensor, 
                       reconstructed: torch.Tensor, 
                       reduction: str = 'mean',
                       loss_type: str = 'mse') -> torch.Tensor:
    """
    Average feature reconstruction error within each sample, then reduce the batch.
    
    Args:
        original: Input features [B, ...].
        reconstructed: Reconstructed features with the same shape.
        reduction: Batch reduction: 'mean', 'sum', or 'none'.
        loss_type: Elementwise loss: 'mse', 'mae', or 'huber'.
        
    Returns:
        A scalar loss, or per-sample losses when reduction is 'none'.
    """
    if loss_type == 'mse':
        loss = F.mse_loss(reconstructed, original, reduction='none')
    elif loss_type == 'mae':
        loss = F.l1_loss(reconstructed, original, reduction='none')
    elif loss_type == 'huber':
        loss = F.huber_loss(reconstructed, original, reduction='none', delta=1.0)
    else:
        raise ValueError(f"Unsupported loss type: {loss_type}")
    
    # Average over non-batch dimensions.
    if loss.dim() > 2:
        loss = loss.mean(dim=tuple(range(1, loss.dim())))
    elif loss.dim() == 2:
        loss = loss.mean(dim=1)
    
    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    else:
        return loss


def sparsity_penalty(activations: torch.Tensor,
                    target_sparsity: Optional[float] = None,
                    penalty_type: str = 'l1',
                    reduction: str = 'mean') -> torch.Tensor:
    """
    Compute an activation penalty with the selected reduction.
    
    Args:
        activations: Activation tensor [B, ...].
        target_sparsity: Target mean activation for the KL penalty.
        penalty_type: One of 'l1', 'l2', or 'kl'.
        reduction: 'mean', 'sum', or 'none'.
        
    Returns:
        Reduced penalty, per-sample L1/L2 penalties, or elementwise KL penalties.
    """
    if penalty_type == 'l1':
        penalty = torch.abs(activations)
    elif penalty_type == 'l2':
        penalty = activations ** 2
    elif penalty_type == 'kl':
        if target_sparsity is None:
            raise ValueError("target_sparsity is required for the KL penalty")
        
        rho = target_sparsity
        rho_hat = torch.mean(activations, dim=0)
        
        # Keep logarithms finite.
        rho_hat = torch.clamp(rho_hat, min=1e-8, max=1-1e-8)
        
        kl_div = rho * torch.log(rho / rho_hat) + (1 - rho) * torch.log((1 - rho) / (1 - rho_hat))
        
        if reduction == 'mean':
            return torch.mean(kl_div)
        elif reduction == 'sum':
            return torch.sum(kl_div)
        else:
            return kl_div
    else:
        raise ValueError(f"Unsupported sparsity penalty: {penalty_type}")
    
    # Reduce feature dimensions before applying the batch reduction.
    if penalty.dim() > 2:
        penalty = penalty.mean(dim=tuple(range(1, penalty.dim())))
    elif penalty.dim() == 2:
        penalty = penalty.mean(dim=1)
    
    if reduction == 'mean':
        return penalty.mean()
    elif reduction == 'sum':
        return penalty.sum()
    else:
        return penalty


def cross_modal_kl_loss(img_sparse: torch.Tensor,
                       txt_sparse: torch.Tensor,
                       temperature: float = 1.0,
                       symmetric: bool = True) -> torch.Tensor:
    """
    Compute KL divergence between softmax-transformed image and text activations.
    
    Args:
        img_sparse: Image activations [B, n].
        txt_sparse: Text activations [B, m].
        temperature: Softmax temperature.
        symmetric: Whether to average both KL directions.
        
    Returns:
        Scalar divergence loss.
    """
    # Bound logits before computing probabilities.
    img_logits = torch.clamp(img_sparse / temperature, min=-20.0, max=20.0)
    txt_logits = torch.clamp(txt_sparse / temperature, min=-20.0, max=20.0)

    img_prob = F.softmax(img_logits, dim=-1)
    txt_prob = F.softmax(txt_logits, dim=-1)
    
    if img_prob.size(-1) != txt_prob.size(-1):
        # Compare the shared prefix when feature dimensions differ.
        min_dim = min(img_prob.size(-1), txt_prob.size(-1))
        img_prob = img_prob[:, :min_dim]
        txt_prob = txt_prob[:, :min_dim]
    
    kl_img_txt = F.kl_div(
        torch.log(img_prob + 1e-8), 
        txt_prob, 
        reduction='batchmean'
    )
    
    if symmetric:
        kl_txt_img = F.kl_div(
            torch.log(txt_prob + 1e-8), 
            img_prob, 
            reduction='batchmean'
        )
        return (kl_img_txt + kl_txt_img) / 2
    else:
        return kl_img_txt


def cross_modal_cosine_loss(img_sparse: torch.Tensor,
                           txt_sparse: torch.Tensor,
                           target_similarity: float = 1.0) -> torch.Tensor:
    """
    Penalize the difference between feature similarity and its target.
    
    Args:
        img_sparse: Image activations [B, n].
        txt_sparse: Text activations [B, m].
        target_similarity: Target dot product of normalized features.
        
    Returns:
        Mean squared similarity error.
    """
    img_norm = F.normalize(img_sparse, p=2, dim=-1)
    txt_norm = F.normalize(txt_sparse, p=2, dim=-1)
    
    # Compare the shared prefix after normalization.
    if img_norm.size(-1) != txt_norm.size(-1):
        min_dim = min(img_norm.size(-1), txt_norm.size(-1))
        img_norm = img_norm[:, :min_dim]
        txt_norm = txt_norm[:, :min_dim]
    
    cosine_sim = torch.sum(img_norm * txt_norm, dim=-1)  # [B]
    
    loss = F.mse_loss(cosine_sim, torch.full_like(cosine_sim, target_similarity))
    
    return loss


def cross_modal_js_loss(img_sparse: torch.Tensor,
                       txt_sparse: torch.Tensor,
                       temperature: float = 1.0) -> torch.Tensor:
    """
    Compute Jensen-Shannon divergence between image and text activations.
    
    Args:
        img_sparse: Image activations [B, n].
        txt_sparse: Text activations [B, m].
        temperature: Softmax temperature.
        
    Returns:
        Scalar divergence loss.
    """
    img_prob = F.softmax(img_sparse / temperature, dim=-1)
    txt_prob = F.softmax(txt_sparse / temperature, dim=-1)
    
    # Compare the shared prefix when feature dimensions differ.
    if img_prob.size(-1) != txt_prob.size(-1):
        min_dim = min(img_prob.size(-1), txt_prob.size(-1))
        img_prob = img_prob[:, :min_dim]
        txt_prob = txt_prob[:, :min_dim]
    
    mean_prob = (img_prob + txt_prob) / 2
    
    # JS(P, Q) = (KL(P || M) + KL(Q || M)) / 2, with M = (P + Q) / 2.
    kl_img_mean = F.kl_div(torch.log(mean_prob + 1e-8), img_prob, reduction='batchmean')
    kl_txt_mean = F.kl_div(torch.log(mean_prob + 1e-8), txt_prob, reduction='batchmean')
    
    js_div = (kl_img_mean + kl_txt_mean) / 2
    
    return js_div

def _ue_uncorr_and_evenness(H: torch.Tensor,
                            center: bool = True,
                            eps: float = 1e-8) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Return uncorrelation and evenness penalties for activations H [B, N].

    Uncorrelation penalizes off-diagonal covariance energy after per-feature
    normalization. A B x B Gram matrix avoids constructing an N x N covariance.
    Evenness penalizes deviations of feature second moments from their mean.
    With centering enabled, these second moments are feature variances.
    """
    if H.dim() != 2:
        H = H.view(H.size(0), -1)
    B, N = H.shape
    assert B >= 1 and N >= 1, "UE regularization requires nonempty batch and feature dimensions"

    Hc = H - H.mean(dim=0, keepdim=True) if center else H

    # Per-feature second moments, used by both penalties.
    var = Hc.pow(2).mean(dim=0) + eps              # [N]
    std = var.sqrt()                                # [N]

    # Normalize each feature before computing the Gram matrix.
    Hn = Hc / std.clamp_min(eps)                    # [B, N]
    # Use a B x B Gram matrix instead of an N x N covariance matrix.
    G = Hn @ Hn.t()                                 # [B, B]
    s = Hn.pow(2).sum(dim=0)                        # [N], diagonal of Hn.T @ Hn

    # For covariance Hn.T @ Hn / (B-1), off-diagonal energy is
    # (||G||_F^2 - ||s||_2^2) / (B-1)^2, averaged over feature pairs below.
    offdiag_sq = (G.pow(2).sum() - s.pow(2).sum())
    denom = (max(B - 1, 1) ** 2) * (N * max(N - 1, 1) + 1e-8)
    L_uncorr = offdiag_sq / denom

    # Penalize unequal feature second moments.
    v_norm = var / (var.mean() + eps)
    L_even = (v_norm - 1.0).pow(2).mean()

    return L_uncorr, L_even



class SAELoss(nn.Module):
    """Weighted reconstruction, sparsity, cross-modal, and UE losses."""
    
    def __init__(self,
                 lambda_rec: float = 1.0,
                 lambda_sp: float = 0.1, 
                 lambda_kl: float = 0.01,
                 reconstruction_type: str = 'mse',
                 sparsity_type: str = 'l1',
                 cross_modal_type: str = 'kl',
                 target_sparsity: float = 0.05,
                 temperature: float = 1.0,
                 lambda_ue: float = 0.0,
                 ue_even_weight: float = 1.0):
        """
        Configure loss components and their weights.
        
        Args:
            lambda_rec: Reconstruction weight.
            lambda_sp: Sparsity penalty weight.
            lambda_kl: Cross-modal loss weight.
            reconstruction_type: Elementwise reconstruction loss.
            sparsity_type: Activation penalty type.
            cross_modal_type: Cross-modal comparison: 'kl', 'cosine', or 'js'.
            target_sparsity: Target mean activation for the KL sparsity penalty.
            temperature: Cross-modal softmax temperature.
            lambda_ue: Uncorrelation and evenness penalty weight.
            ue_even_weight: Relative weight of the evenness term.
        """
        super().__init__()
        
        self.lambda_rec = lambda_rec
        self.lambda_sp = lambda_sp  
        self.lambda_kl = lambda_kl
        self.reconstruction_type = reconstruction_type
        self.sparsity_type = sparsity_type
        self.cross_modal_type = cross_modal_type
        self.target_sparsity = target_sparsity
        self.temperature = temperature
        self.lambda_ue = lambda_ue
        self.ue_even_weight = ue_even_weight

        print(f"SAELoss initialized:")
        print(f"  Reconstruction weight: {lambda_rec}, type: {reconstruction_type}")
        print(f"  Sparsity weight: {lambda_sp}, type: {sparsity_type}")
        if lambda_sp == 0:
            print(f"    Sparsity penalty disabled")
        print(f"  Cross-modal weight: {lambda_kl}, type: {cross_modal_type}")
        if lambda_kl == 0:
            print(f"    Cross-modal loss disabled")
        if self.lambda_ue > 0:
            print(f"  UE regularization enabled: weight={self.lambda_ue}, even_w={self.ue_even_weight}")

    def forward(self, 
                img_outputs: Dict[str, torch.Tensor],
                txt_outputs: Dict[str, torch.Tensor],
                return_components: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict]]:
        """
        Compute the weighted loss and optional component statistics.
        
        Args:
            img_outputs: Image SAE output dictionary.
            txt_outputs: Text SAE output dictionary.
            return_components: Whether to return individual loss values.
            
        Returns:
            Total loss, optionally paired with a component dictionary.
        """
        img_rec_loss = reconstruction_loss(
            img_outputs['input'], 
            img_outputs['reconstructed'],
            loss_type=self.reconstruction_type
        )
        
        txt_rec_loss = reconstruction_loss(
            txt_outputs['input'],
            txt_outputs['reconstructed'], 
            loss_type=self.reconstruction_type
        )
        
        L_rec = img_rec_loss + txt_rec_loss
        
        # Evaluate optional penalties only when their weights are positive.
        if self.lambda_sp > 0:
            img_sp_loss = sparsity_penalty(
                img_outputs['sparse_activations'],
                target_sparsity=self.target_sparsity,
                penalty_type=self.sparsity_type
            )
            
            txt_sp_loss = sparsity_penalty(
                txt_outputs['sparse_activations'],
                target_sparsity=self.target_sparsity, 
                penalty_type=self.sparsity_type
            )
            
            L_sp = img_sp_loss + txt_sp_loss
        else:
            img_sp_loss = torch.tensor(0.0, device=L_rec.device)
            txt_sp_loss = torch.tensor(0.0, device=L_rec.device)
            L_sp = torch.tensor(0.0, device=L_rec.device)
        
        if self.lambda_kl > 0:
            if self.cross_modal_type == 'kl':
                L_kl = cross_modal_kl_loss(
                    img_outputs['global_sparse'],
                    txt_outputs['global_sparse'],
                    temperature=self.temperature
                )
            elif self.cross_modal_type == 'cosine':
                L_kl = cross_modal_cosine_loss(
                    img_outputs['global_sparse'],
                    txt_outputs['global_sparse']
                )
            elif self.cross_modal_type == 'js':
                L_kl = cross_modal_js_loss(
                    img_outputs['global_sparse'],
                    txt_outputs['global_sparse'],
                    temperature=self.temperature
                )
            else:
                raise ValueError(f"Unsupported cross-modal loss: {self.cross_modal_type}")
        else:
            L_kl = torch.tensor(0.0, device=L_rec.device)

        if self.lambda_ue > 0:
            # Compute UE penalties on one aggregated vector per sample.
            img_unc, img_even = _ue_uncorr_and_evenness(img_outputs['global_sparse'])
            txt_unc, txt_even = _ue_uncorr_and_evenness(txt_outputs['global_sparse'])
            L_ue_core = (img_unc + txt_unc) + self.ue_even_weight * (img_even + txt_even)
        else:
            L_ue_core = torch.tensor(0.0, device=L_rec.device)

        total_loss = (self.lambda_rec * L_rec +
                      self.lambda_sp * L_sp +
                      self.lambda_kl * L_kl +
                      self.lambda_ue * L_ue_core)
        
        if return_components:
            components = {
                'total_loss': total_loss.item(),
                'reconstruction_loss': L_rec.item(),
                'sparsity_loss': L_sp.item(), 
                'cross_modal_loss': L_kl.item(),
                'img_rec_loss': img_rec_loss.item(),
                'txt_rec_loss': txt_rec_loss.item(),
                'img_sp_loss': img_sp_loss.item(),
                'txt_sp_loss': txt_sp_loss.item(),
                'ue_loss': (L_ue_core.item() if self.lambda_ue > 0 else 0.0)
            }
            return total_loss, components
        else:
            return total_loss


def compute_activation_statistics(img_sparse: torch.Tensor,
                                txt_sparse: torch.Tensor) -> Dict[str, float]:
    """Summarize image and text activations and their active-dimension overlap."""
    with torch.no_grad():
        stats = {}
        
        img_nonzero_ratio = (img_sparse != 0).float().mean().item()
        img_mean_activation = torch.mean(torch.abs(img_sparse)).item()
        img_active_dims = (img_sparse != 0).sum(dim=-1).float().mean().item()
        
        txt_nonzero_ratio = (txt_sparse != 0).float().mean().item()
        txt_mean_activation = torch.mean(torch.abs(txt_sparse)).item()
        txt_active_dims = (txt_sparse != 0).sum(dim=-1).float().mean().item()
        
        # Measure overlap when both modalities have the same feature dimension.
        if img_sparse.size(-1) == txt_sparse.size(-1):
            img_active = (img_sparse != 0).float()
            txt_active = (txt_sparse != 0).float()
            overlap = torch.sum(img_active * txt_active, dim=-1).mean().item()
            union = torch.sum((img_active + txt_active) > 0, dim=-1).float().mean().item()
            jaccard = overlap / (union + 1e-8)
            
            stats['activation_overlap'] = overlap
            stats['activation_jaccard'] = jaccard
        
        stats.update({
            'img_nonzero_ratio': img_nonzero_ratio,
            'img_mean_activation': img_mean_activation,
            'img_active_dims': img_active_dims,
            'txt_nonzero_ratio': txt_nonzero_ratio,
            'txt_mean_activation': txt_mean_activation,
            'txt_active_dims': txt_active_dims,
            'active_dims_diff': abs(img_active_dims - txt_active_dims)
        })
        
        return stats


# Loss presets.
LOSS_CONFIGS = {
    'default': {
        'lambda_rec': 1.0,
        'lambda_sp': 0.1,
        'lambda_kl': 0.01,
        'reconstruction_type': 'mse',
        'sparsity_type': 'l1',
        'cross_modal_type': 'kl',
        'target_sparsity': 0.05,
        'temperature': 1.0
    },
    'strong_sparsity': {
        'lambda_rec': 1.0,
        'lambda_sp': 0.5,
        'lambda_kl': 0.01,
        'reconstruction_type': 'mse',
        'sparsity_type': 'l1',
        'cross_modal_type': 'kl',
        'target_sparsity': 0.02,
        'temperature': 1.0
    },
    'strong_alignment': {
        'lambda_rec': 1.0,
        'lambda_sp': 0.05,
        'lambda_kl': 0.1,
        'reconstruction_type': 'mse',
        'sparsity_type': 'l1',
        'cross_modal_type': 'js',
        'target_sparsity': 0.05,
        'temperature': 0.5
    }
} 
