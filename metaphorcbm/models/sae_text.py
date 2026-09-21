"""Transformer text encoders, sparse autoencoders, and concept labeling utilities."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, Tuple, Optional, List
from .sparsity import KSparseMask, TopKActivation, sparsity_loss


class TransformerEncoder(nn.Module):
    """Transformer encoder with learned token and position embeddings."""
    
    def __init__(self,
                 vocab_size: int,
                 d_model: int = 512,
                 n_heads: int = 8,
                 n_layers: int = 6,
                 d_ff: int = 3072,
                 max_length: int = 40,
                 dropout: float = 0.1):
        """
        Configure the text encoder.
        
        Args:
            vocab_size: Number of tokens in the vocabulary.
            d_model: Token embedding dimension.
            n_heads: Number of attention heads.
            n_layers: Number of encoder layers.
            d_ff: Feedforward hidden dimension.
            max_length: Number of learned position embeddings.
            dropout: Dropout probability.
        """
        super().__init__()
        
        self.d_model = d_model
        self.max_length = max_length
        
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_length, d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize linear and embedding weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)
    
    def load_pretrained(self, ckpt_path: str, strict: bool = True):
        """Load a transformer state dictionary with optional strict key matching."""
        state = torch.load(ckpt_path, map_location='cpu')
        missing, unexpected = self.load_state_dict(state, strict=strict)
        print(f"[TransformerEncoder] Pretrained weights loaded")
        if missing:
            print(f"  Missing keys: {missing}")
        if unexpected:
            print(f"  Unexpected keys: {unexpected}")
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Encode a batch of token sequences.
        
        Args:
            input_ids: Token IDs [B, seq_len].
            attention_mask: Valid-token mask [B, seq_len], with zero for padding.
            
        Returns:
            Contextual token features [B, seq_len, d_model].
        """
        batch_size, seq_len = input_ids.shape
        
        position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        
        token_embeds = self.token_embedding(input_ids)
        pos_embeds = self.position_embedding(position_ids)
        embeddings = token_embeds + pos_embeds
        embeddings = self.dropout(embeddings)
        
        # PyTorch's padding mask uses True for positions to ignore.
        padding_mask = (attention_mask == 0)
        
        encoded = self.transformer(embeddings, src_key_padding_mask=padding_mask)
        encoded = self.layer_norm(encoded)
        
        return encoded


class TextSAE(nn.Module):
    """Encode token features into sparse activations and reconstruct them."""
    
    def __init__(self,
                 d_text: int = 512,
                 hidden_dim: int = 2048,
                 k_sparse: int = 16,
                 use_bias: bool = False,
                 activation: str = 'relu',
                 initialization: str = 'fan_in'):
        """
        Initialize the text encoder, sparsity mask, and decoder.
        
        Args:
            d_text: Input token feature dimension.
            hidden_dim: Number of latent features.
            k_sparse: Rank used for the activation threshold.
            use_bias: Whether the encoder uses a bias term.
            activation: Activation function before sparsification.
            initialization: Weight initialization method.
        """
        super().__init__()
        
        self.d_text = d_text
        self.hidden_dim = hidden_dim
        self.k_sparse = k_sparse
        
        self.encoder = nn.Sequential(
            nn.Linear(d_text, hidden_dim, bias=use_bias)
        )
        
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        else:
            self.activation = nn.ReLU()
            
        self.k_sparse_mask = KSparseMask(k=k_sparse, dim=-1)
        
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, d_text)
        )
        
        self._initialize_weights(initialization)
        
        print(f"TextSAE initialized:")
        print(f"  Input dimension: {d_text}")
        print(f"  Hidden dimension: {hidden_dim} (expansion: {hidden_dim/d_text:.1f})")
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
            x: Token features [B, seq_len, d_text] or pooled features [B, d_text].
            
        Returns:
            Sparse activations with the last dimension replaced by hidden_dim.
        """
        h = self.encoder(x)
        
        h = self.activation(h)
        
        sparse_h = self.k_sparse_mask(h)
        
        return sparse_h
    
    def decode(self, sparse_h: torch.Tensor) -> torch.Tensor:
        """Reconstruct token features from sparse activations."""
        return self.decoder(sparse_h)
    
    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Reconstruct token features and sum their concept activations.
        
        Args:
            x: Token features [B, seq_len, d_text] or pooled features [B, d_text].
            attention_mask: Optional valid-token mask [B, seq_len].
            
        Returns:
            Sparse activations, reconstruction, summed activations, and input.
        """
        original_shape = x.shape
        
        # Encode each token independently: [B, seq_len, d_text] -> [B*seq_len, d_text].
        if x.dim() == 3:
            B, seq_len, feat_dim = x.shape
            x_flat = x.reshape(-1, feat_dim)
            is_sequence = True
        else:
            B, feat_dim = x.shape
            x_flat = x
            seq_len = 1
            is_sequence = False
        
        sparse_activations = self.encode(x_flat)
        
        reconstructed = self.decode(sparse_activations)
        
        if is_sequence:
            sparse_activations = sparse_activations.reshape(B, seq_len, self.hidden_dim)
            reconstructed = reconstructed.reshape(B, seq_len, feat_dim)
            
            # Exclude padding when aggregating sequence-level activations.
            if attention_mask is not None:
                mask_expanded = attention_mask.unsqueeze(-1).float()  # [B, seq_len, 1]
                global_sparse = torch.sum(sparse_activations * mask_expanded, dim=1)  # [B, hidden_dim]
            else:
                global_sparse = torch.sum(sparse_activations, dim=1)  # [B, hidden_dim]
        else:
            global_sparse = sparse_activations  # [B, hidden_dim]
        
        return {
            'sparse_activations': sparse_activations,
            'reconstructed': reconstructed,
            'global_sparse': global_sparse,
            'input': x.reshape(original_shape)
        }

    def get_decoder_weights(self) -> torch.Tensor:
        """Return decoder weights [d_text, hidden_dim] with gradients attached."""
        return self.decoder[-1].weight


class TransformerWithHooks(nn.Module):
    """Wrap a transformer to expose token features and optional pooled features."""
    
    def __init__(self,
                 vocab_size: int,
                 d_model: int = 512,
                 n_heads: int = 8,
                 n_layers: int = 6,
                 max_length: int = 40,
                 dropout: float = 0.1,
                 freeze_backbone: bool = True,
                 return_features: bool = True):
        """
        Configure the transformer feature extractor.
        
        Args:
            vocab_size: Number of tokens in the vocabulary.
            d_model: Token feature dimension.
            n_heads: Number of attention heads.
            n_layers: Number of encoder layers.
            max_length: Number of learned position embeddings.
            dropout: Dropout probability.
            freeze_backbone: Whether to freeze transformer parameters.
            return_features: Whether to include pooled features in the output.
        """
        super().__init__()
        
        self.backbone = TransformerEncoder(
            vocab_size=vocab_size,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            d_ff=d_model * 4,
            max_length=max_length,
            dropout=dropout
        )
        
        self.freeze_backbone = freeze_backbone
        self.return_features = return_features
        self.d_model = d_model
        
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        print(f"TransformerWithHooks initialized:")
        print(f"  Vocabulary size: {vocab_size}")
        print(f"  Model dimension: {d_model}")
        print(f"  Backbone frozen: {freeze_backbone}")
    
    def load_pretrained(self, ckpt_path: str, strict: bool = True):
        """Load transformer weights and reapply the backbone freezing setting."""
        self.backbone.load_pretrained(ckpt_path, strict)
        
        if self.freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Extract token features and optionally their masked average.
        
        Args:
            input_ids: Token IDs [B, seq_len].
            attention_mask: Valid-token mask [B, seq_len].
            
        Returns:
            Sequence features and optional pooled features.
        """
        sequence_features = self.backbone(input_ids, attention_mask)  # [B, seq_len, d_model]
        
        result = {
            'sequence_features': sequence_features,  # [B, seq_len, d_model]
        }
        
        if self.return_features:
            if attention_mask is not None:
                # Average over valid tokens only.
                mask_expanded = attention_mask.unsqueeze(-1).float()  # [B, seq_len, 1]
                valid_features = sequence_features * mask_expanded
                pooled_features = valid_features.sum(dim=1) / mask_expanded.sum(dim=1)  # [B, d_model]
            else:
                pooled_features = sequence_features.mean(dim=1)  # [B, d_model]
            
            result['pooled_features'] = pooled_features
        
        return result


class JointTextModel(nn.Module):
    """Combine a transformer encoder with a text SAE."""
    
    def __init__(self,
                 vocab_size: int,
                 d_model: int = 512,
                 sae_hidden_dim: int = 2048,
                 k_sparse: int = 16,
                 n_heads: int = 8,
                 n_layers: int = 6,
                 max_length: int = 40,
                 dropout: float = 0.1):
        """
        Configure the transformer and sparse autoencoder.
        
        Args:
            vocab_size: Number of tokens in the vocabulary.
            d_model: Token feature dimension.
            sae_hidden_dim: Number of SAE latent features.
            k_sparse: Rank used for the activation threshold.
            n_heads: Number of attention heads.
            n_layers: Number of encoder layers.
            max_length: Number of learned position embeddings.
            dropout: Dropout probability.
        """
        super().__init__()
        
        self.transformer = TransformerEncoder(
            vocab_size=vocab_size,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            d_ff=d_model * 4,
            max_length=max_length,
            dropout=dropout
        )
        
        self.sae = TextSAE(
            d_text=d_model,
            hidden_dim=sae_hidden_dim,
            k_sparse=k_sparse
        )
        
        print(f"JointTextModel initialized:")
        print(f"  Vocabulary size: {vocab_size}")
        print(f"  Transformer dimension: {d_model}")
        print(f"  SAE hidden dimension: {sae_hidden_dim}")
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Encode tokens and reconstruct their features with the SAE.
        
        Args:
            input_ids: Token IDs [B, seq_len].
            attention_mask: Valid-token mask [B, seq_len].
            
        Returns:
            Transformer features and SAE outputs in one dictionary.
        """
        transformer_output = self.transformer(input_ids, attention_mask)  # [B, seq_len, d_model]
        
        sae_output = self.sae(transformer_output, attention_mask)
        
        return {
            'transformer_output': transformer_output,
            **sae_output
        }


class ConceptNamer:
    """Label SAE dimensions using activating tokens or dimension identifiers."""
    
    def __init__(self, text_model: JointTextModel, tokenizer):
        """Use a trained text model and its tokenizer to label concepts."""
        self.text_model = text_model
        self.tokenizer = tokenizer
        self.text_model.eval()
    
    @torch.no_grad()
    def extract_concept_labels_by_activation(self, 
                                           dataloader, 
                                           concept_dim: int,
                                           top_k: int = 10) -> List[str]:
        """
        Decode tokens with the largest activation for a concept.
        
        Args:
            dataloader: Batches containing token IDs and attention masks.
            concept_dim: SAE dimension to inspect.
            top_k: Number of activating token occurrences to decode.
            
        Returns:
            Decoded token strings ordered by activation.
        """
        activations = []
        all_tokens = []
        
        for batch in dataloader:
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            
            outputs = self.text_model(input_ids, attention_mask)
            sparse_acts = outputs['sparse_activations']  # [B, seq_len, hidden_dim]
            
            concept_acts = sparse_acts[:, :, concept_dim]  # [B, seq_len]
            
            concept_acts_flat = concept_acts.reshape(-1)
            input_ids_flat = input_ids.reshape(-1)
            attention_mask_flat = attention_mask.reshape(-1)
            
            # Ignore padding positions when collecting candidate tokens.
            valid_mask = attention_mask_flat.bool()
            activations.extend(concept_acts_flat[valid_mask].tolist())
            all_tokens.extend(input_ids_flat[valid_mask].tolist())
        
        activations = torch.tensor(activations)
        all_tokens = torch.tensor(all_tokens)
        
        _, top_indices = torch.topk(activations, min(top_k, len(activations)))
        top_tokens = all_tokens[top_indices]
        
        labels = [self.tokenizer.decode([token.item()], skip_special_tokens=True).strip() 
                 for token in top_tokens]
        
        return labels
    
    @torch.no_grad()
    def extract_concept_labels_by_generation(self, concept_dim: int) -> str:
        """Decode a one-hot activation and return its dimension identifier."""
        hidden_dim = self.text_model.sae.hidden_dim
        one_hot = torch.zeros(1, 1, hidden_dim)  # [1, 1, hidden_dim]
        one_hot[0, 0, concept_dim] = 1.0
        
        decoded_embedding = self.text_model.sae.decode(one_hot)  # [1, 1, d_model]
        
        return f"concept_{concept_dim}"
    
    def name_all_concepts(self, dataloader, method: str = 'activation') -> Dict[int, str]:
        """
        Assign a label to each SAE dimension.
        
        Args:
            dataloader: Batches used to collect activating tokens.
            method: 'activation' for token labels or 'generation' for identifiers.
            
        Returns:
            Mapping from dimension indices to labels.
        """
        hidden_dim = self.text_model.sae.hidden_dim
        concept_names = {}
        
        if method == 'activation':
            for dim in range(hidden_dim):
                if dim % 100 == 0:
                    print(f"Labeling concept dimension {dim}/{hidden_dim}")
                
                labels = self.extract_concept_labels_by_activation(dataloader, dim, top_k=3)
                concept_names[dim] = "_".join(labels[:3]) if labels else f"concept_{dim}"
        
        elif method == 'generation':
            for dim in range(hidden_dim):
                concept_names[dim] = self.extract_concept_labels_by_generation(dim)
        
        return concept_names


def create_text_sae(config: Optional[Dict] = None) -> TextSAE:
    """Build a text SAE from a configuration dictionary."""
    config = config or {}
    
    return TextSAE(
        d_text=config.get('d_text', 512),
        hidden_dim=config.get('hidden_dim', 2048),
        k_sparse=config.get('k_sparse', 16),
        use_bias=config.get('use_bias', False),
        activation=config.get('activation', 'relu'),
        initialization=config.get('initialization', 'fan_in')
    )


def create_transformer_with_hooks(vocab_size: int, config: Optional[Dict] = None) -> TransformerWithHooks:
    """Build a transformer feature extractor for the given vocabulary size."""
    config = config or {}
    
    return TransformerWithHooks(
        vocab_size=vocab_size,
        d_model=config.get('d_model', 512),
        n_heads=config.get('n_heads', 8),
        n_layers=config.get('n_layers', 6),
        max_length=config.get('max_length', 40),
        dropout=config.get('dropout', 0.1),
        freeze_backbone=config.get('freeze_backbone', True),
        return_features=config.get('return_features', True)
    )


def create_text_sae(config: Optional[Dict] = None) -> TextSAE:
    """Build a text SAE using d_model and sae_hidden_dim configuration fields."""
    config = config or {}
    
    return TextSAE(
        d_text=config.get('d_model', 512),
        hidden_dim=config.get('sae_hidden_dim', 2048),
        k_sparse=config.get('k_sparse', 16),
        use_bias=config.get('use_bias', False),
        activation=config.get('activation', 'relu'),
        initialization=config.get('initialization', 'fan_in')
    )


def create_joint_text_model(vocab_size: int, config: Optional[Dict] = None) -> JointTextModel:
    """Build a transformer and text SAE for the given vocabulary size."""
    config = config or {}
    
    return JointTextModel(
        vocab_size=vocab_size,
        d_model=config.get('d_model', 512),
        sae_hidden_dim=config.get('sae_hidden_dim', 2048),
        k_sparse=config.get('k_sparse', 16),
        n_heads=config.get('n_heads', 12),
        n_layers=config.get('n_layers', 6),
        max_length=config.get('max_length', 40),
        dropout=config.get('dropout', 0.1)
    )


# Text model presets.
TEXT_SAE_CONFIGS = {
    'default': {
        'd_text': 512,
        'hidden_dim': 2048,
        'k_sparse': 16,
        'use_bias': False,
        'activation': 'relu',
        'initialization': 'fan_in'
    },
    'large': {
        'd_text': 512,
        'hidden_dim': 6144,
        'k_sparse': 20,
        'use_bias': False,
        'activation': 'gelu',
        'initialization': 'xavier'
    }
}

JOINT_TEXT_CONFIGS = {
    'default': {
        'd_model': 512,
        'sae_hidden_dim': 2048,
        'k_sparse': 16,
        'n_heads': 8,
        'n_layers': 6,
        'max_length': 40,
        'dropout': 0.1
    }
} 
