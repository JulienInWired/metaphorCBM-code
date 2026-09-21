"""Dataset utilities for deterministic concept coupling extraction."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from metaphorcbm.data import COCODataset, ImageTransforms, TextTransforms


@dataclass
class PairEntry:
    image_path: str
    image_id: int
    caption: str


class DeterministicCaptionDataset(Dataset):
    """Wrap COCODataset to expose deterministic image-caption pairs."""

    def __init__(
        self,
        root_dir: str,
        ann_file: str,
        image_transforms,
        text_transforms,
        split: str = "train",
        max_samples: Optional[int] = None,
        seed: int = 42,
        balanced_sampling: bool = False,
        caption_mode: str = "first",
    ):
        self.base = COCODataset(
            root_dir=root_dir,
            ann_file=ann_file,
            image_transforms=image_transforms,
            text_transforms=text_transforms,
            split=split,
            max_samples=max_samples,
            seed=seed,
            balanced_sampling=balanced_sampling,
        )
        self.root_dir = root_dir
        self.ann_file = ann_file
        self.split = split
        self.image_transforms = image_transforms
        self.text_transforms = text_transforms
        self.caption_mode = caption_mode
        self.seed = seed
        self.entries = self._expand_entries()

    def _expand_entries(self) -> List[PairEntry]:
        rng = random.Random(self.seed)
        entries: List[PairEntry] = []
        for pair in self.base.pairs:
            captions = list(pair["captions"])
            if not captions:
                continue
            if self.caption_mode == "all":
                chosen = captions
            elif self.caption_mode == "random_fixed":
                chosen = [rng.choice(captions)]
            else:
                chosen = [captions[0]]
            for caption in chosen:
                entries.append(PairEntry(image_path=pair["image_path"], image_id=int(pair["image_id"]), caption=caption))
        return entries

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        entry = self.entries[idx]
        image = Image.open(entry.image_path)
        if image.mode != "RGB":
            image = image.convert("RGB")
        if self.image_transforms is not None:
            image = self.image_transforms(image)
        else:
            image = torch.tensor([[0.0]])
        text_data = self.text_transforms(entry.caption) if self.text_transforms is not None else {"text": entry.caption}
        payload: Dict[str, object] = {
            "image": image,
            "caption": entry.caption,
            "image_id": entry.image_id,
            "image_path": entry.image_path,
        }
        payload.update(text_data)
        return payload

    def get_image_captions(self, image_id: int) -> List[str]:
        return self.base.get_image_captions(image_id)



def build_dataloader(
    *,
    root_dir: str,
    ann_file: str,
    image_transforms,
    text_transforms,
    split: str,
    max_samples: Optional[int],
    seed: int,
    balanced_sampling: bool,
    caption_mode: str,
    batch_size: int,
    num_workers: int,
) -> DataLoader:
    dataset = DeterministicCaptionDataset(
        root_dir=root_dir,
        ann_file=ann_file,
        image_transforms=image_transforms,
        text_transforms=text_transforms,
        split=split,
        max_samples=max_samples,
        seed=seed,
        balanced_sampling=balanced_sampling,
        caption_mode=caption_mode,
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
