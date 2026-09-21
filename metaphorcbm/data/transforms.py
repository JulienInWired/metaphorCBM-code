"""Image resizing and normalization, and fixed-length text tokenization."""

import torch
import torch.nn as nn
from torchvision import transforms
from transformers import AutoTokenizer
from PIL import Image
import numpy as np
from typing import Dict, List, Tuple, Optional


class ImageTransforms:
    """Resize, center-crop, and normalize images."""
    
    def __init__(self, 
                 resize_size: int = 256,
                 crop_size: int = 224,
                 imagenet_normalize: bool = True):
        """
        Configure image preprocessing.
        
        Args:
            resize_size: Length of the resized image's shorter side.
            crop_size: Side length of the square center crop.
            imagenet_normalize: Use ImageNet statistics, or 0.5 for mean and scale.
        """
        self.resize_size = resize_size
        self.crop_size = crop_size
        
        if imagenet_normalize:
            self.mean = [0.485, 0.456, 0.406]
            self.std = [0.229, 0.224, 0.225]
        else:
            self.mean = [0.5, 0.5, 0.5]
            self.std = [0.5, 0.5, 0.5]
        
        self.train_transform = transforms.Compose([
            transforms.Resize(self.resize_size),
            transforms.CenterCrop(self.crop_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std)
        ])
        
        self.val_transform = transforms.Compose([
            transforms.Resize(self.resize_size),
            transforms.CenterCrop(self.crop_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std)
        ])
    
    def __call__(self, image: Image.Image, is_training: bool = True) -> torch.Tensor:
        """Convert a PIL image to a normalized tensor [3, crop_size, crop_size]."""
        if not isinstance(image, Image.Image):
            raise TypeError("Expected a PIL.Image input")
        
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        transform = self.train_transform if is_training else self.val_transform
        return transform(image)
    
    def denormalize(self, tensor: torch.Tensor) -> torch.Tensor:
        """Undo channel normalization for a [3, H, W] tensor."""
        mean = torch.tensor(self.mean).view(3, 1, 1)
        std = torch.tensor(self.std).view(3, 1, 1)
        return tensor * std + mean


class TextTransforms:
    """Tokenize text with padding and truncation to a fixed length."""
    
    def __init__(self, 
                 tokenizer_name: str = "bert-base-uncased",
                 max_length: int = 40,
                 d_text: int = 512,
                 add_special_tokens: bool = True):
        """
        Configure the tokenizer.
        
        Args:
            tokenizer_name: Hugging Face tokenizer name or local directory.
            max_length: Length of the padded or truncated token sequence.
            d_text: Stored text feature dimension.
            add_special_tokens: Whether to add the tokenizer's special tokens.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length
        self.d_text = d_text
        self.add_special_tokens = add_special_tokens
        
        # Use the end-of-sequence token if no padding token is defined.
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def __call__(self, text: str) -> Dict[str, torch.Tensor]:
        """Return input_ids and attention_mask tensors, each of shape [max_length]."""
        if not isinstance(text, str):
            raise TypeError("Expected a string input")
        
        encoded = self.tokenizer(
            text,
            add_special_tokens=self.add_special_tokens,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Remove the tokenizer's batch axis; the data loader adds it later.
        return {
            'input_ids': encoded['input_ids'].squeeze(0),
            'attention_mask': encoded['attention_mask'].squeeze(0)
        }
    
    def batch_encode(self, texts: List[str]) -> Dict[str, torch.Tensor]:
        """Return batched input_ids and attention_mask tensors [B, max_length]."""
        encoded = self.tokenizer(
            texts,
            add_special_tokens=self.add_special_tokens,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoded['input_ids'],
            'attention_mask': encoded['attention_mask']
        }
    
    def decode(self, token_ids: torch.Tensor, skip_special_tokens: bool = True) -> str:
        """Decode token IDs, optionally omitting special tokens."""
        return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)
    
    def get_vocab_size(self) -> int:
        """Return the tokenizer vocabulary size."""
        return len(self.tokenizer)


def create_transforms(config: Optional[Dict] = None) -> Tuple[ImageTransforms, TextTransforms]:
    """Build image and text transforms from their configuration sections."""
    config = config or {}
    
    image_config = config.get('image', {})
    text_config = config.get('text', {})
    
    image_transforms = ImageTransforms(
        resize_size=image_config.get('resize_size', 256),
        crop_size=image_config.get('crop_size', 224),
        imagenet_normalize=image_config.get('imagenet_normalize', True)
    )
    
    text_transforms = TextTransforms(
        tokenizer_name=text_config.get('tokenizer_name', 'bert-base-uncased'),
        max_length=text_config.get('max_length', 40),
        d_text=text_config.get('d_text', 512),
        add_special_tokens=text_config.get('add_special_tokens', True)
    )
    
    return image_transforms, text_transforms


# Preprocessing defaults.
DEFAULT_CONFIG = {
    'image': {
        'resize_size': 256,
        'crop_size': 224,
        'imagenet_normalize': True
    },
    'text': {
        'tokenizer_name': 'bert-base-uncased',
        'max_length': 40,
        'd_text': 512,
        'add_special_tokens': True
    }
} 
