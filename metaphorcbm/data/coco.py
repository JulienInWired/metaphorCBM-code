"""Load images and randomly sampled captions from COCO-format annotations."""

import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import random
from typing import Dict, List, Tuple, Optional, Union
from .transforms import ImageTransforms, TextTransforms


class COCODataset(Dataset):
    """Image-caption pairs with optional category-balanced subsampling."""
    
    def __init__(self,
                 root_dir: str,
                 ann_file: str,
                 image_transforms: Optional[ImageTransforms] = None,
                 text_transforms: Optional[TextTransforms] = None,
                 split: str = 'train',
                 max_samples: Optional[int] = None,
                 seed: int = 42,
                 balanced_sampling: bool = False):
        """
        Load annotations and index images with available captions.
        
        Args:
            root_dir: Directory containing train2017/ and val2017/.
            ann_file: COCO-format caption annotation JSON.
            image_transforms: Image preprocessing callable.
            text_transforms: Caption preprocessing callable.
            split: Image split, 'train' or 'val'.
            max_samples: Maximum number of images; None uses all images.
            seed: Python random seed.
            balanced_sampling: Balance subsampling using instance annotations.
        """
        self.root_dir = root_dir
        self.ann_file = ann_file
        self.split = split
        self.image_transforms = image_transforms
        self.text_transforms = text_transforms
        self.max_samples = max_samples
        self.balanced_sampling = balanced_sampling
        
        random.seed(seed)
        
        self._load_annotations()
        
        if self.balanced_sampling:
            self._load_category_info()
        
        self._build_image_caption_pairs()
    
    def _load_annotations(self):
        """Index image metadata and captions by image ID."""
        if not os.path.exists(self.ann_file):
            raise FileNotFoundError(f"Annotation file not found: {self.ann_file}")
        
        print(f"Loading annotations: {self.ann_file}")
        with open(self.ann_file, 'r', encoding='utf-8') as f:
            self.coco_data = json.load(f)
        
        self.images = {img['id']: img for img in self.coco_data['images']}
        
        self.img_to_captions = {}
        for ann in self.coco_data['annotations']:
            img_id = ann['image_id']
            if img_id not in self.img_to_captions:
                self.img_to_captions[img_id] = []
            self.img_to_captions[img_id].append(ann['caption'].strip())
        
        print(f"Loaded {len(self.images)} images and {len(self.coco_data['annotations'])} captions")
    
    def _load_category_info(self):
        """Load instance categories for balanced subsampling."""
        instances_file = self.ann_file.replace('captions_', 'instances_')
        
        if not os.path.exists(instances_file):
            print(f"Warning: Instance annotations not found at {instances_file}; using random sampling")
            self.balanced_sampling = False
            return
        
        print(f"Loading categories: {instances_file}")
        with open(instances_file, 'r', encoding='utf-8') as f:
            instances_data = json.load(f)
        
        self.categories = {cat['id']: cat['name'] for cat in instances_data['categories']}
        
        self.img_to_categories = {}
        for ann in instances_data['annotations']:
            img_id = ann['image_id']
            cat_id = ann['category_id']
            if img_id not in self.img_to_categories:
                self.img_to_categories[img_id] = set()
            self.img_to_categories[img_id].add(cat_id)
        
        for img_id in self.img_to_categories:
            self.img_to_categories[img_id] = list(self.img_to_categories[img_id])
        
        print(f"Loaded {len(self.categories)} categories for {len(self.img_to_categories)} annotated images")
    
    def _build_image_caption_pairs(self):
        """Index existing images and optionally subsample them."""
        self.pairs = []
        
        for img_id, img_info in self.images.items():
            if img_id in self.img_to_captions:
                captions = self.img_to_captions[img_id]
                
                # Keep all captions together under one image entry.
                img_path = os.path.join(self.root_dir, 
                                      f"{self.split}2017", 
                                      img_info['file_name'])
                
                if os.path.exists(img_path):
                    self.pairs.append({
                        'image_path': img_path,
                        'captions': captions,
                        'image_id': img_id
                    })
        
        if self.max_samples is not None and self.max_samples < len(self.pairs):
            if self.balanced_sampling and hasattr(self, 'img_to_categories'):
                self.pairs = self._balanced_sample(self.pairs, self.max_samples)
            else:
                self.pairs = random.sample(self.pairs, self.max_samples)
        
        print(f"Indexed {len(self.pairs)} images with captions")
    
    def _balanced_sample(self, pairs: List[Dict], max_samples: int) -> List[Dict]:
        """Allocate the sample budget across randomly assigned primary categories."""
        from collections import defaultdict
        
        category_groups = defaultdict(list)
        
        for pair in pairs:
            img_id = pair['image_id']
            if img_id in self.img_to_categories:
                # Assign multi-category images to one randomly chosen category.
                primary_category = random.choice(self.img_to_categories[img_id])
                category_groups[primary_category].append(pair)
            else:
                # Reserve a group for images without instance categories.
                category_groups[-1].append(pair)
        
        num_categories = len(category_groups)
        samples_per_category = max_samples // num_categories
        remaining_samples = max_samples % num_categories
        
        balanced_pairs = []
        category_names = []
        
        for cat_id, cat_pairs in category_groups.items():
            current_samples = samples_per_category
            if remaining_samples > 0:
                current_samples += 1
                remaining_samples -= 1
            
            if current_samples >= len(cat_pairs):
                balanced_pairs.extend(cat_pairs)
                actual_samples = len(cat_pairs)
            else:
                selected_pairs = random.sample(cat_pairs, current_samples)
                balanced_pairs.extend(selected_pairs)
                actual_samples = current_samples
            
            if cat_id == -1:
                cat_name = "Unannotated"
            else:
                cat_name = self.categories.get(cat_id, f"Category_{cat_id}")
            category_names.append(f"{cat_name}: {actual_samples}")
        
        print(f"Balanced sample: {len(balanced_pairs)} images")
        print(f"Category counts: {', '.join(category_names)}")
        
        random.shuffle(balanced_pairs)
        
        return balanced_pairs
    
    def __len__(self) -> int:
        """Return the number of indexed images."""
        return len(self.pairs)
    
    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, str, int]]:
        """Load an image, sample one caption, and apply the configured transforms."""
        pair = self.pairs[idx]
        
        try:
            image = Image.open(pair['image_path'])
            if image.mode != 'RGB':
                image = image.convert('RGB')
        except Exception as e:
            print(f"Unable to load image {pair['image_path']}: {e}")
            # Substitute a black image when loading fails.
            image = Image.new('RGB', (224, 224), color='black')
        
        caption = random.choice(pair['captions'])
        
        if self.image_transforms is not None:
            image = self.image_transforms(image)
        else:
            image = torch.tensor([[0.0]])  # Placeholder when no transform is provided.
        
        if self.text_transforms is not None:
            text_data = self.text_transforms(caption)
        else:
            text_data = {'text': caption}
        
        return {
            'image': image,
            'caption': caption,
            'image_id': pair['image_id'],
            'image_path': pair['image_path'],
            **text_data
        }
    
    def get_image_captions(self, image_id: int) -> List[str]:
        """Return all captions for an image ID."""
        return self.img_to_captions.get(image_id, [])
    
    def get_random_sample(self) -> Dict:
        """Return one randomly selected sample."""
        idx = random.randint(0, len(self) - 1)
        return self[idx]


def create_coco_dataloaders(
    root_dir: str,
    train_ann_file: str,
    val_ann_file: str,
    image_transforms: ImageTransforms,
    text_transforms: TextTransforms,
    batch_size: int = 64,
    num_workers: int = 8,
    train_max_samples: Optional[int] = None,
    val_max_samples: Optional[int] = None
) -> Tuple[DataLoader, DataLoader]:
    """
    Create training and validation loaders for COCO-format image-caption data.
    
    Args:
        root_dir: Directory containing the image splits.
        train_ann_file: Training caption annotations.
        val_ann_file: Validation caption annotations.
        image_transforms: Image preprocessing callable.
        text_transforms: Caption preprocessing callable.
        batch_size: Number of images per batch.
        num_workers: Data-loading worker count.
        train_max_samples: Optional training image limit.
        val_max_samples: Optional validation image limit.
        
    Returns:
        (train_dataloader, val_dataloader)
    """
    
    train_dataset = COCODataset(
        root_dir=root_dir,
        ann_file=train_ann_file,
        image_transforms=image_transforms,
        text_transforms=text_transforms,
        split='train',
        max_samples=train_max_samples
    )
    
    val_dataset = COCODataset(
        root_dir=root_dir,
        ann_file=val_ann_file,
        image_transforms=image_transforms,
        text_transforms=text_transforms,
        split='val',
        max_samples=val_max_samples
    )
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=4,
        drop_last=True
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )
    
    return train_dataloader, val_dataloader


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Stack images and optional token tensors, retaining captions and image IDs."""
    images = torch.stack([item['image'] for item in batch])
    captions = [item['caption'] for item in batch]
    image_ids = torch.tensor([item['image_id'] for item in batch])
    
    if 'input_ids' in batch[0]:
        input_ids = torch.stack([item['input_ids'] for item in batch])
        attention_mask = torch.stack([item['attention_mask'] for item in batch])
        
        return {
            'images': images,
            'captions': captions,
            'image_ids': image_ids,
            'input_ids': input_ids,
            'attention_mask': attention_mask
        }
    else:
        return {
            'images': images,
            'captions': captions,
            'image_ids': image_ids
        }
