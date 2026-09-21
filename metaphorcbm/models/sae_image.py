"""Sparse autoencoders for spatial and pooled image features."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional
from .sparsity import KSparseMask, TopKActivation, sparsity_loss


class ImageSAE(nn.Module):
    """Encode image features into sparse activations and reconstruct the input."""
    
    def __init__(self,
                 input_dim: int = 2048,
                 hidden_dim: int = 8192, 
                 k_sparse: int = 15,
                 use_bias: bool = False,
                 activation: str = 'relu',
                 initialization: str = 'fan_in'):
        """
        Initialize the encoder, sparsity mask, and decoder.
        
        Args:
            input_dim: Number of input features.
            hidden_dim: Number of latent features.
            k_sparse: Rank used for the activation threshold.
            use_bias: Whether the encoder uses a bias term.
            activation: Activation function before sparsification.
            initialization: Weight initialization method.
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.k_sparse = k_sparse
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=use_bias)
        )
        
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        else:
            self.activation = nn.ReLU()
            
        self.k_sparse_mask = KSparseMask(k=k_sparse, dim=-1)
        
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, input_dim)
        )
        
        self._initialize_weights(initialization)
        
        print(f"ImageSAE initialized:")
        print(f"  Input dimension: {input_dim}")
        print(f"  Hidden dimension: {hidden_dim} (expansion: {hidden_dim/input_dim:.1f})")
        print(f"  k-sparse: {k_sparse} (active fraction: {k_sparse/hidden_dim:.3f})")
    
    def _initialize_weights(self, method: str = 'fan_in'):
        """Initialize linear weights and zero their biases."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                if method == 'fan_in':
                    nn.init.kaiming_uniform_(module.weight, mode='fan_in', nonlinearity='relu')
                elif method == 'xavier':
                    nn.init.xavier_uniform_(module.weight)
                elif method == 'normal':
                    nn.init.normal_(module.weight, std=0.01)
                
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode features along the last dimension.
        
        Args:
            x: Features of shape [B, spatial_dim, input_dim] or [B, input_dim].
            
        Returns:
            Sparse activations with the last dimension replaced by hidden_dim.
        """
        h = self.encoder(x)
        
        h = self.activation(h)
        sparse_h = self.k_sparse_mask(h)
        
        return sparse_h
    
    def decode(self, sparse_h: torch.Tensor) -> torch.Tensor:
        """Reconstruct input features from sparse activations."""
        return self.decoder(sparse_h)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Reconstruct features and aggregate spatial concept activations.
        
        Args:
            x: Features of shape [B, spatial_dim, input_dim] or [B, input_dim].
            
        Returns:
            Sparse activations, reconstruction, spatially summed activations,
            and the input features.
        """
        original_shape = x.shape
        
        # Encode spatial locations as independent feature vectors.
        if x.dim() == 3:
            B, spatial_dim, feat_dim = x.shape
            x_flat = x.reshape(-1, feat_dim)
            is_spatial = True
        else:
            B, feat_dim = x.shape
            x_flat = x
            spatial_dim = 1
            is_spatial = False
        
        sparse_activations = self.encode(x_flat)

        reconstructed = self.decode(sparse_activations)
        
        if is_spatial:
            sparse_activations = sparse_activations.reshape(B, spatial_dim, self.hidden_dim)
            reconstructed = reconstructed.reshape(B, spatial_dim, feat_dim)
            
            # Sum over spatial locations to obtain image-level activations.
            global_sparse = torch.sum(sparse_activations, dim=1)  # [B, hidden_dim]
        else:
            global_sparse = sparse_activations  # [B, hidden_dim]
        
        return {
            'sparse_activations': sparse_activations,
            'reconstructed': reconstructed, 
            'global_sparse': global_sparse,
            'input': x.reshape(original_shape)
        }
    
    def get_sparsity_stats(self, sparse_activations: torch.Tensor) -> Dict[str, float]:
        """Summarize activation density, magnitude, and active feature counts."""
        with torch.no_grad():
            nonzero_ratio = (sparse_activations != 0).float().mean().item()
            
            mean_activation = torch.mean(torch.abs(sparse_activations)).item()
            
            active_dims_per_sample = (sparse_activations != 0).sum(dim=-1).float().mean().item()
            
            return {
                'nonzero_ratio': nonzero_ratio,
                'mean_activation': mean_activation,
                'active_dims_per_sample': active_dims_per_sample,
                'target_active_dims': self.k_sparse
            }

    def get_decoder_weights(self) -> torch.Tensor:
        """Return decoder weights [input_dim, hidden_dim] with gradients attached."""
        return self.decoder[-1].weight


class ConceptBottleneckImageSAE(nn.Module):
    """Image SAE with concept names and activation statistics."""
    
    def __init__(self,
                 input_dim: int = 2048,
                 hidden_dim: int = 8192,
                 k_sparse: int = 15,
                 concept_names: Optional[list] = None):
        super().__init__()
        
        self.sae = ImageSAE(input_dim, hidden_dim, k_sparse)
        self.concept_names = concept_names or [f"concept_{i}" for i in range(hidden_dim)]
        
        self.register_buffer('concept_counts', torch.zeros(hidden_dim))
        self.register_buffer('total_samples', torch.tensor(0))
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Run the SAE and update concept counts during training."""
        outputs = self.sae(x)
        
        if self.training:
            self._update_concept_stats(outputs['global_sparse'])
        
        return outputs
    
    def _update_concept_stats(self, global_sparse: torch.Tensor):
        """Count images with a nonzero activation for each concept."""
        batch_size = global_sparse.size(0)
        
        active_concepts = (global_sparse != 0).sum(dim=0)  # [hidden_dim]
        self.concept_counts += active_concepts
        self.total_samples += batch_size
    
    def get_concept_activation_rates(self) -> torch.Tensor:
        """Return the fraction of images activating each concept."""
        return self.concept_counts / (self.total_samples + 1e-8)
    
    def get_top_concepts(self, k: int = 10) -> list:
        """Return the k most frequently active concepts and their rates."""
        rates = self.get_concept_activation_rates()
        _, top_indices = torch.topk(rates, k)
        
        return [(self.concept_names[i], rates[i].item()) for i in top_indices]
    
    def reset_concept_stats(self):
        """Reset concept counts and the sample counter."""
        self.concept_counts.zero_()
        self.total_samples.zero_()


def create_image_sae(config: Optional[Dict] = None) -> ImageSAE:
    """Build an image SAE from a configuration dictionary."""
    config = config or {}
    
    return ImageSAE(
        input_dim=config.get('input_dim', 2048),
        hidden_dim=config.get('hidden_dim', 8192),
        k_sparse=config.get('k_sparse', 15),
        use_bias=config.get('use_bias', False),
        activation=config.get('activation', 'relu'),
        initialization=config.get('initialization', 'fan_in')
    )


# Image SAE presets.
IMAGE_SAE_CONFIGS = {
    'default': {
        'input_dim': 2048,
        'hidden_dim': 8192,
        'k_sparse': 15,
        'use_bias': False,
        'activation': 'relu',
        'initialization': 'fan_in'
    },
    'compact': {
        'input_dim': 2048,
        'hidden_dim': 4096,
        'k_sparse': 10,
        'use_bias': False,
        'activation': 'relu',
        'initialization': 'fan_in'
    },
    'large': {
        'input_dim': 2048,
        'hidden_dim': 16384,
        'k_sparse': 20,
        'use_bias': False,
        'activation': 'gelu',
        'initialization': 'xavier'
    }
} 
