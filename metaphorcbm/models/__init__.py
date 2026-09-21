"""Visual backbones, sparse autoencoders, and sparsity operators."""

from .resnet_hooks import ResNetWithHooks, create_clip_resnet_backbone
from .sae_image import ImageSAE
from .sae_text import TextSAE, TransformerWithHooks, JointTextModel, create_transformer_with_hooks, create_text_sae, create_joint_text_model
from .sparsity import KSparseMask, sparsity_loss

__all__ = ['ResNetWithHooks', 'create_clip_resnet_backbone', 'ImageSAE', 'TextSAE', 'TransformerWithHooks', 'JointTextModel', 'create_transformer_with_hooks', 'create_text_sae', 'create_joint_text_model', 'KSparseMask', 'sparsity_loss']
