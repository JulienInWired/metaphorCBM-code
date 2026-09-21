"""Extract spatial and pooled features from ResNet backbones using forward hooks."""

import torch
import torch.nn as nn
import torchvision.models as models
from typing import Dict, List, Optional, Callable
import warnings
import clip as openai_clip  # type: ignore

class ResNetWithHooks(nn.Module):
    """Torchvision ResNet-50 with a configurable feature hook."""
    
    def __init__(self, 
                 weights: str = "IMAGENET1K_V2",
                 hook_layer: str = "layer4",
                 freeze_backbone: bool = True,
                 return_features: bool = True):
        """
        Load ResNet-50 and attach a feature hook.
        
        Args:
            weights: Torchvision pretrained weights identifier.
            hook_layer: Layer from which to capture features.
            freeze_backbone: Whether to freeze backbone parameters.
            return_features: Whether to include classification logits in the output.
        """
        super().__init__()
        
        self.backbone = models.resnet50(weights=weights)
        self.hook_layer = hook_layer
        self.return_features = return_features
        
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        self.features = {}
        self.hooks = []
        
        self._register_hooks()
        
        print(f"ResNet-50 loaded with weights: {weights}")
        print(f"Hook layer: {hook_layer}, frozen: {freeze_backbone}")
    
    def _register_hooks(self):
        """Capture the selected layer's output during each forward pass."""
        
        def hook_fn(name):
            def hook(module, input, output):
                self.features[name] = output
            return hook
        
        target_layer = getattr(self.backbone, self.hook_layer)
        handle = target_layer.register_forward_hook(hook_fn(self.hook_layer))
        self.hooks.append(handle)
        
        print(f"Forward hook registered on {self.hook_layer}")
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Extract spatial features and their global average.
        
        Args:
            x: Image batch [B, 3, H, W].
            
        Returns:
            Spatial, pooled, and raw features, with optional classification logits.
        """
        self.features.clear()
        
        logits = self.backbone(x)
        
        if self.hook_layer in self.features:
            layer4_features = self.features[self.hook_layer]
            
            # Flatten the spatial grid: [B, C, H, W] -> [B, H*W, C].
            B, C, H, W = layer4_features.shape
            spatial_features = layer4_features.permute(0, 2, 3, 1).reshape(B, H*W, C)
            
            result = {
                'spatial_features': spatial_features,  # [B, H*W, C]
                'pooled_features': layer4_features.mean(dim=[2, 3]),  # [B, C]
                'raw_features': layer4_features,  # [B, C, H, W]
            }
            
            if self.return_features:
                result['logits'] = logits  # [B, 1000]
            
            return result
        else:
            warnings.warn(f"No features captured from layer {self.hook_layer}")
            return {'logits': logits}
    
    def get_feature_shapes(self) -> Dict[str, tuple]:
        """Inspect output shapes with a single 224 x 224 image."""
        dummy_input = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            outputs = self.forward(dummy_input)
        
        shapes = {}
        for key, value in outputs.items():
            if isinstance(value, torch.Tensor):
                shapes[key] = tuple(value.shape)
        
        return shapes
    
    def remove_hooks(self):
        """Remove all registered forward hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
        print("All forward hooks removed")
    
    def __del__(self):
        """Release forward hooks when the wrapper is destroyed."""
        self.remove_hooks()

class CLIPResNetWithHooks(nn.Module):
    """
    OpenAI CLIP ResNet with spatial feature hooks.

    Output keys match ResNetWithHooks. For RN50 layer4 at 224 x 224 resolution,
    spatial features have shape [B, 49, 2048] and pooled features [B, 2048].
    The optional 'logits' field contains CLIP image embeddings [B, 1024].

    Args:
        clip_variant: Model name passed to OpenAI CLIP.
        pretrained: Weight-source label used in status messages.
        hook_layer: Layer from which to capture features.
        freeze_backbone: Whether to freeze the visual encoder.
        return_features: Whether to include image embeddings as 'logits'.
        finetuned_model_path: Optional checkpoint overriding visual encoder weights.
    """
    def __init__(self,
                 clip_variant: str = "RN50",
                 pretrained: str = "openai",
                 hook_layer: str = "layer4",
                 freeze_backbone: bool = True,
                 return_features: bool = True,
                 finetuned_model_path: Optional[str] = None):
        super().__init__()
        self.clip_variant = clip_variant
        self.pretrained = pretrained
        self.hook_layer = hook_layer
        self.return_features = return_features

        self.features: Dict[str, torch.Tensor] = {}
        self.hooks: List[torch.utils.hooks.RemovableHandle] = []
        self.preprocess = None   # Preprocessing returned by the CLIP loader.
        self.clip_model = None
        self.backbone = None
        self._backend = None

        try:
            self.clip_model, self.preprocess = openai_clip.load(self.clip_variant, jit=False)
            self.clip_model = self.clip_model.float()
            self.backbone = self.clip_model.visual
            self._backend = "openai/clip"
        except Exception as e_openai_clip:
            raise ImportError(
                "Failed to load the OpenAI CLIP model."
            ) from e_openai_clip

        if freeze_backbone and self.backbone is not None:
            for p in self.backbone.parameters():
                p.requires_grad = False

        if finetuned_model_path is not None:
            self._load_finetuned_weights(finetuned_model_path)

        self._register_hooks()

        model_desc = f"Fine-tuned weights ({finetuned_model_path})" if finetuned_model_path else f"Pretrained: {self.pretrained}"
        print(f"CLIP {self.clip_variant} loaded via {self._backend}, {model_desc}")
        print(f"Hook layer: {hook_layer}, frozen: {freeze_backbone}")

    def _resolve_target_layer(self):
        """Find the hook layer on the backbone or its nested trunk."""
        if self.backbone is None:
            raise RuntimeError("The CLIP visual encoder is not initialized.")
        if hasattr(self.backbone, self.hook_layer):
            return getattr(self.backbone, self.hook_layer)
        if hasattr(self.backbone, "trunk") and hasattr(self.backbone.trunk, self.hook_layer):
            return getattr(self.backbone.trunk, self.hook_layer)
        raise AttributeError(f"Layer {self.hook_layer} not found in the CLIP visual encoder.")

    def _register_hooks(self):
        """Capture features from the selected visual layer."""
        def hook_fn(name):
            def _hook(module, inp, out):
                self.features[name] = out
            return _hook

        target_layer = self._resolve_target_layer()
        handle = target_layer.register_forward_hook(hook_fn(self.hook_layer))
        self.hooks.append(handle)
        print(f"Forward hook registered on {self.hook_layer} (CLIP)")

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Return spatial features, pooled features, and optional image embeddings.

        Image embeddings are stored under 'logits' without extra normalization.
        """
        self.features.clear()

        if hasattr(self.clip_model, "encode_image"):
            clip_embedding = self.clip_model.encode_image(x)
        else:
            clip_embedding = self.backbone(x)

        if self.hook_layer in self.features:
            layer4 = self.features[self.hook_layer]           # [B, C, H, W]
            B, C, H, W = layer4.shape
            spatial = layer4.permute(0, 2, 3, 1).reshape(B, H * W, C)  # [B, HW, C]
            result = {
                "spatial_features": spatial,                  # [B, H*W, C]
                "pooled_features": layer4.mean(dim=[2, 3]),  # [B, C]
                "raw_features": layer4,                       # [B, C, H, W]
            }
            if self.return_features:
                result["logits"] = clip_embedding             # [B, embedding_dim]
            return result
        else:
            warnings.warn(f"No features captured from layer {self.hook_layer} (CLIP)")
            return {"logits": clip_embedding}

    def _load_finetuned_weights(self, finetuned_model_path: str):
        """Load visual encoder weights with the 'backbone.' prefix removed."""
        try:
            print(f"Loading fine-tuned weights: {finetuned_model_path}")
            checkpoint = torch.load(finetuned_model_path, map_location='cpu')
            
            if 'model_state_dict' in checkpoint:
                model_state_dict = checkpoint['model_state_dict']
            else:
                model_state_dict = checkpoint
            
            backbone_state_dict = {}
            backbone_params = 0
            for key, value in model_state_dict.items():
                if key.startswith('backbone.'):
                    new_key = key[9:]  # Strip 'backbone.'.
                    backbone_state_dict[new_key] = value
                    backbone_params += value.numel()
            
            if not backbone_state_dict:
                raise ValueError("No backbone weights found in the fine-tuned checkpoint")
            
            missing_keys, unexpected_keys = self.backbone.load_state_dict(backbone_state_dict, strict=False)
            
            if missing_keys:
                print(f"Warning: Missing weights: {missing_keys[:5]}{'...' if len(missing_keys) > 5 else ''} ({len(missing_keys)} total)")
            if unexpected_keys:
                print(f"Warning: Unused weights: {unexpected_keys[:5]}{'...' if len(unexpected_keys) > 5 else ''} ({len(unexpected_keys)} total)")
            
            print(f"Fine-tuned weights loaded: {len(backbone_state_dict)} tensors, {backbone_params:,} parameters")
            
        except Exception as e:
            raise RuntimeError(f"Failed to load fine-tuned weights: {e}")

    def remove_hooks(self):
        """Remove all registered forward hooks."""
        for h in self.hooks:
            h.remove()
        self.hooks.clear()
        print("All forward hooks removed (CLIP)")

    def __del__(self):
        self.remove_hooks()

class MultiLayerResNetHooks(nn.Module):
    """Extract features from multiple ResNet layers."""
    
    def __init__(self, 
                 weights: str = "IMAGENET1K_V2",
                 hook_layers: List[str] = ["layer3", "layer4"],
                 freeze_backbone: bool = True):
        """
        Load ResNet-50 and attach hooks to the selected layers.
        
        Args:
            weights: Torchvision pretrained weights identifier.
            hook_layers: Names of layers to capture.
            freeze_backbone: Whether to freeze backbone parameters.
        """
        super().__init__()
        
        self.backbone = models.resnet50(weights=weights)
        self.hook_layers = hook_layers
        
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        self.features = {}
        self.hooks = []
        
        self._register_multi_hooks()
    
    def _register_multi_hooks(self):
        """Register one forward hook per available layer."""
        
        def make_hook(name):
            def hook(module, input, output):
                self.features[name] = output
            return hook
        
        for layer_name in self.hook_layers:
            if hasattr(self.backbone, layer_name):
                layer = getattr(self.backbone, layer_name)
                handle = layer.register_forward_hook(make_hook(layer_name))
                self.hooks.append(handle)
                print(f"Forward hook registered on {layer_name}")
            else:
                print(f"Warning: Layer {layer_name} not found")
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Return logits and spatial, pooled, and raw features for each layer."""
        self.features.clear()
        logits = self.backbone(x)
        
        result = {'logits': logits}
        
        for layer_name in self.hook_layers:
            if layer_name in self.features:
                features = self.features[layer_name]
                B, C, H, W = features.shape
                
                result[f'{layer_name}_spatial'] = features.permute(0, 2, 3, 1).reshape(B, H*W, C)
                result[f'{layer_name}_pooled'] = features.mean(dim=[2, 3])
                result[f'{layer_name}_raw'] = features
        
        return result
    
    def remove_hooks(self):
        """Remove all registered forward hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()


def create_resnet_backbone(config: Optional[Dict] = None) -> ResNetWithHooks:
    """Build a hooked ResNet backbone from a configuration dictionary."""
    config = config or {}
    
    return ResNetWithHooks(
        weights=config.get('weights', 'IMAGENET1K_V2'),
        hook_layer=config.get('hook_layer', 'layer4'),
        freeze_backbone=config.get('freeze_backbone', True),
        return_features=config.get('return_features', True)
    )

def create_clip_resnet_backbone(config: Optional[Dict] = None) -> CLIPResNetWithHooks:
    """
    Build a hooked CLIP ResNet backbone from a configuration dictionary.

    Example:
        model = create_clip_resnet_backbone({
            'clip_variant': 'RN50',
            'pretrained': 'openai',
            'hook_layer': 'layer4',
            'freeze_backbone': True,
            'return_features': True,
            'finetuned_model_path': 'path/to/best_model.pth'  # Optional override.
        })
    """
    config = config or {}
    return CLIPResNetWithHooks(
        clip_variant=config.get("clip_variant", "RN50"),
        pretrained=config.get("pretrained", "openai"),
        hook_layer=config.get("hook_layer", "layer4"),
        freeze_backbone=config.get("freeze_backbone", True),
        return_features=config.get("return_features", True),
        finetuned_model_path=config.get("finetuned_model_path", None),
    )

class FeatureExtractor:
    """Collect backbone features from images or data loaders."""
    
    def __init__(self, model: ResNetWithHooks):
        self.model = model
        self.model.eval()
    
    @torch.no_grad()
    def extract_features(self, dataloader) -> Dict[str, List[torch.Tensor]]:
        """Return concatenated CPU feature tensors and their image IDs."""
        all_features = {
            'spatial_features': [],
            'pooled_features': [],
            'image_ids': []
        }
        
        for batch in dataloader:
            images = batch['image']
            image_ids = batch.get('image_id', [])
            
            outputs = self.model(images)
            
            all_features['spatial_features'].append(outputs['spatial_features'].cpu())
            all_features['pooled_features'].append(outputs['pooled_features'].cpu())
            if image_ids:
                all_features['image_ids'].extend(image_ids.tolist())
        
        all_features['spatial_features'] = torch.cat(all_features['spatial_features'], dim=0)
        all_features['pooled_features'] = torch.cat(all_features['pooled_features'], dim=0)
        
        return all_features
    
    def get_single_image_features(self, image: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Extract features, adding a batch dimension for a single image."""
        if image.dim() == 3:
            image = image.unsqueeze(0)
        
        with torch.no_grad():
            return self.model(image)


# ResNet feature dimensions for a 224 x 224 input.
RESNET_LAYER_INFO = {
    'conv1': {'channels': 64, 'spatial': (112, 112)},
    'layer1': {'channels': 256, 'spatial': (56, 56)},
    'layer2': {'channels': 512, 'spatial': (28, 28)},
    'layer3': {'channels': 1024, 'spatial': (14, 14)},
    'layer4': {'channels': 2048, 'spatial': (7, 7)},
}


def print_resnet_info():
    """Print channel counts and spatial dimensions for ResNet-50 layers."""
    print("ResNet-50 layer dimensions:")
    print("-" * 50)
    for layer, info in RESNET_LAYER_INFO.items():
        channels = info['channels']
        h, w = info['spatial']
        print(f"{layer:8s}: {channels:4d} channels, {h:3d}×{w:3d} spatial")
    print("-" * 50) 
