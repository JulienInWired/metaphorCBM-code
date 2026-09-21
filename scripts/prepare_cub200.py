# -*- coding: utf-8 -*-
"""
Prepare CUB-200-2011 images and captions in ImageFolder and COCO formats.

Images are organized into train/ and test/. COCO output adds train2017/ and
val2017/ links plus annotations/captions_train2017.json and captions_val2017.json.
Caption sources include the ten descriptions per image released by Reed et al.
(CVPR 2016) and the AttnGAN preprocessing archive.

Google Drive downloads use gdown with a urllib fallback. When directory links
are unavailable, image directories are copied, requiring additional disk space.
"""

import os
import sys
import io
import json
import tarfile
import zipfile
import shutil
import argparse
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from tqdm import tqdm


# Download and filesystem utilities.

def _human_bytes(n: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    x = float(n)
    while x >= 1024 and i < len(units) - 1:
        x /= 1024.0
        i += 1
    return f"{x:.1f}{units[i]}"


def _download_with_progress(url: str, dst_path: Path):
    """Download an HTTP resource with byte and percentage progress."""
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    def reporthook(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            percent = min(100.0, downloaded * 100.0 / total_size)
            msg = f"\rDownloading: {percent:5.1f}% ({_human_bytes(downloaded)}/{_human_bytes(total_size)})"
        else:
            msg = f"\rDownloaded: {_human_bytes(downloaded)}"
        sys.stdout.write(msg)
        sys.stdout.flush()

    urllib.request.urlretrieve(url, str(dst_path), reporthook=reporthook)
    print("\nDownload complete:", dst_path)


def _download_from_gdrive(file_id: str, dst_path: Path) -> bool:
    """Download from Google Drive using gdown or a direct URL; return success."""
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import gdown  # type: ignore
        url = f"https://drive.google.com/uc?id={file_id}"
        print(f"Downloading with gdown: {url}")
        gdown.download(url, str(dst_path), quiet=False)
        ok = dst_path.exists() and dst_path.stat().st_size > 0
        if ok:
            print("Google Drive download complete:", dst_path)
        return ok
    except Exception as e:
        print(f"Warning: gdown unavailable or download failed: {e}")
        print("Trying a direct urllib download; Google Drive may require confirmation...")
        try:
            url = f"https://drive.google.com/uc?export=download&id={file_id}"
            _download_with_progress(url, dst_path)
            return dst_path.exists() and dst_path.stat().st_size > 0
        except Exception as e2:
            print("Direct download failed:", e2)
            print("Install gdown with `pip install gdown` and retry.")
            return False


def _extract_any(archive_path: Path, target_dir: Path):
    """Extract a tar, gzip-compressed tar, or ZIP archive into target_dir."""
    target_dir.mkdir(parents=True, exist_ok=True)
    suffix = "".join(archive_path.suffixes).lower()
    print(f"Extracting {archive_path.name} -> {target_dir} ...")

    if suffix.endswith(".tgz") or suffix.endswith(".tar.gz") or suffix.endswith(".tar"):
        mode = "r:gz" if suffix.endswith((".tgz", ".tar.gz")) else "r"
        with tarfile.open(archive_path, mode) as tar:
            tar.extractall(target_dir)
    elif suffix.endswith(".zip"):
        with zipfile.ZipFile(archive_path, "r") as zf:
            zf.extractall(target_dir)
    else:
        raise ValueError(f"Unsupported archive format: {suffix}")

    print("Extraction complete")


def _safe_symlink_or_copy(src: Path, dst: Path):
    """Create a relative directory link, copying the directory if linking fails."""
    if dst.exists():
        return
    try:
        link_target = os.path.relpath(src, start=dst.parent)
        os.symlink(link_target, dst, target_is_directory=True)
        print(f"Directory link created: {dst} -> {src}")
    except Exception as e:
        print(f"Warning: Linking failed ({e}); copying the directory instead...")
        shutil.copytree(src, dst)
        print(f"Directory copied: {dst}")


class CUB200Preprocessor:
    """Prepare CUB images, class metadata, and caption annotations."""

    def __init__(self,
                 data_root: str = "./data",
                 coco_compat: bool = True,
                 captions_source: str = "reed_cvpr16"):
        """
        Args:
            data_root: Root directory for downloaded and processed data.
            coco_compat: Create COCO-style image directories and caption JSON files.
            captions_source: 'reed_cvpr16' or 'attngan_preproc'.
        """
        self.data_root = Path(data_root)
        self.data_root.mkdir(parents=True, exist_ok=True)

        self.raw_dir = self.data_root / "cub200_raw"
        self.processed_dir = self.data_root / "cub200"

        # Original CUB-200-2011 image archive.
        self.dataset_url = "https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz?download=1"
        self.dataset_filename = "CUB_200_2011.tgz"

        # Caption archives: Reed et al. (CVPR 2016) and AttnGAN preprocessing.
        self.captions_gdrive_ids = {
            "reed_cvpr16": "0B0ywwgffWnLLZW9uVHNjb2JmNlE",
            "attngan_preproc": "1O_LtUP9sch09QH3s_EBAgLEctBQ5JBSJ",
        }
        if captions_source not in self.captions_gdrive_ids:
            raise ValueError(f"Unsupported captions_source: {captions_source}")
        self.captions_source = captions_source

        self.coco_compat = coco_compat

        self._metadata_df: Optional[pd.DataFrame] = None
        self._classes_df: Optional[pd.DataFrame] = None
        self._captions_index: Dict[str, List[str]] = {}

    # Downloads and extraction.

    def download_dataset(self) -> bool:
        """Download the image archive unless it already exists."""
        dataset_path = self.data_root / self.dataset_filename
        if dataset_path.exists():
            print(f"Dataset archive already exists: {dataset_path}")
            return True

        print(f"Downloading CUB-200...")
        print(f"URL: {self.dataset_url}")
        print(f"Destination: {dataset_path}")
        try:
            _download_with_progress(self.dataset_url, dataset_path)
            return True
        except Exception as e:
            print(f"Download failed: {e}")
            if dataset_path.exists():
                dataset_path.unlink(missing_ok=True)
            return False

    def extract_dataset(self) -> bool:
        """Extract the image archive into cub200_raw/."""
        dataset_path = self.data_root / self.dataset_filename
        if not dataset_path.exists():
            print(f"Dataset archive not found: {dataset_path}")
            return False

        if self.raw_dir.exists():
            print(f"Raw data directory already exists: {self.raw_dir}")
            return True

        try:
            _extract_any(dataset_path, self.data_root)
            extracted_dir = self.data_root / "CUB_200_2011"
            if extracted_dir.exists():
                extracted_dir.rename(self.raw_dir)
            print("CUB image extraction complete")
            return True
        except Exception as e:
            print(f"Extraction failed: {e}")
            return False

    def download_captions(self) -> Optional[Path]:
        """Download the caption archive and return its local path."""
        file_id = self.captions_gdrive_ids[self.captions_source]
        out_path = self.data_root / f"cub_captions_{self.captions_source}.zip"
        if out_path.exists() and out_path.stat().st_size > 0:
            print(f"Caption archive already exists: {out_path}")
            return out_path

        print(f"Downloading captions ({self.captions_source})...")
        ok = _download_from_gdrive(file_id, out_path)
        if ok:
            return out_path
        return None

    def extract_captions(self, captions_zip: Path) -> Optional[Path]:
        """Extract captions and locate the root of their class-organized text files."""
        dest_dir = self.data_root / f"cub200_captions_{self.captions_source}"
        if dest_dir.exists():
            print(f"Captions already extracted: {dest_dir}")
        else:
            _extract_any(captions_zip, dest_dir)

        # Prefer the named text_c10 or text directory.
        candidate_dirs = []
        for p in dest_dir.rglob("*"):
            if p.is_dir() and p.name.lower() in {"text_c10", "text"}:
                candidate_dirs.append(p)

        if not candidate_dirs:
            # Otherwise search for a subtree with more than 1,000 text files.
            for p in dest_dir.rglob("*"):
                if p.is_dir():
                    txt_cnt = len(list(p.glob("**/*.txt")))
                    if txt_cnt > 1000:
                        candidate_dirs.append(p)
                        break

        if not candidate_dirs:
            print("Caption text directory not found. Check the archive layout for text_c10 or text.")
            return None

        captions_text_root = sorted(candidate_dirs, key=lambda x: x.name.lower() != "text_c10")[0]
        print(f"Caption text directory: {captions_text_root}")
        return captions_text_root

    # Image metadata and directory layout.

    def parse_metadata(self) -> Dict[str, pd.DataFrame]:
        """Join image paths, labels, splits, and class names into data frames."""
        print("Reading metadata...")

        images_file = self.raw_dir / "images.txt"
        labels_file = self.raw_dir / "image_class_labels.txt"
        split_file = self.raw_dir / "train_test_split.txt"
        classes_file = self.raw_dir / "classes.txt"

        for fp in [images_file, labels_file, split_file, classes_file]:
            if not fp.exists():
                raise FileNotFoundError(f"Metadata file not found: {fp}")

        images_df = pd.read_csv(images_file, sep=r"\s+", names=["img_id", "filepath"])
        labels_df = pd.read_csv(labels_file, sep=r"\s+", names=["img_id", "target"])
        split_df = pd.read_csv(split_file, sep=r"\s+", names=["img_id", "is_training_img"])
        classes_df = pd.read_csv(classes_file, sep=r"\s+", names=["class_id", "class_name"])

        meta = images_df.merge(labels_df, on="img_id")
        meta = meta.merge(split_df, on="img_id")
        meta = meta.merge(classes_df, left_on="target", right_on="class_id")

        self._metadata_df = meta
        self._classes_df = classes_df

        print("Metadata loaded:")
        print(f"  Images: {len(meta)}")
        print(f"  Training images: {len(meta[meta['is_training_img'] == 1])}")
        print(f"  Test images: {len(meta[meta['is_training_img'] == 0])}")
        print(f"  Classes: {len(classes_df)}")

        return {"metadata": meta, "classes": classes_df}

    def _print_folder_stats(self, folder_path: Path, split_name: str):
        class_dirs = [d for d in folder_path.iterdir() if d.is_dir()]
        total_images = sum(len(list(d.rglob("*.jpg"))) for d in class_dirs)
        print(f"{split_name} statistics:")
        print(f"  Classes: {len(class_dirs)}")
        print(f"  Images: {total_images}")
        if len(class_dirs) > 0:
            print(f"  Images per class: {total_images / len(class_dirs):.1f}")

    def create_imagefolder_structure(self, metadata: Dict[str, pd.DataFrame]):
        """
        Create the ImageFolder layout:
          processed_dir/train/<class_name>/*.jpg
          processed_dir/test/<class_name>/*.jpg
        With coco_compat enabled, also link or copy:
          processed_dir/train2017 -> train
          processed_dir/val2017   -> test
        """
        print("Creating the ImageFolder layout...")

        df = metadata["metadata"]
        classes = metadata["classes"]

        train_dir = self.processed_dir / "train"
        test_dir = self.processed_dir / "test"
        train_dir.mkdir(parents=True, exist_ok=True)
        test_dir.mkdir(parents=True, exist_ok=True)

        for _, row in classes.iterrows():
            (train_dir / row["class_name"]).mkdir(parents=True, exist_ok=True)
            (test_dir / row["class_name"]).mkdir(parents=True, exist_ok=True)

        print("Copying images into train/test splits...")
        pbar = tqdm(df.itertuples(index=False), total=len(df), desc="Copying images")
        for r in pbar:
            src_path = self.raw_dir / "images" / r.filepath
            dst_subdir = train_dir if r.is_training_img == 1 else test_dir
            dst_path = dst_subdir / r.class_name / Path(r.filepath).name
            if not src_path.exists():
                print(f"Warning: Source image not found; skipping {src_path}")
                continue
            if not dst_path.exists():
                shutil.copy2(src_path, dst_path)

        print("ImageFolder layout created")
        self._print_folder_stats(train_dir, "Training split")
        self._print_folder_stats(test_dir, "Test split")

        if self.coco_compat:
            (self.processed_dir / "annotations").mkdir(parents=True, exist_ok=True)
            _safe_symlink_or_copy(train_dir, self.processed_dir / "train2017")
            _safe_symlink_or_copy(test_dir, self.processed_dir / "val2017")

    # Caption indexing and export.

    def _load_captions_index(self, captions_text_root: Path) -> Dict[str, List[str]]:
        """
        Map paths relative to raw_dir/images to lists of nonempty captions.

        For example, '001.Class/IMG.jpg' maps to lines in '001.Class/IMG.txt'.
        """
        assert self._metadata_df is not None
        idx: Dict[str, List[str]] = {}

        total = len(self._metadata_df)
        miss = 0

        for r in tqdm(self._metadata_df.itertuples(index=False), total=total, desc="Indexing captions"):
            rel = Path(r.filepath)                  # 001.Class/xxx.jpg
            txt_path = captions_text_root / rel.with_suffix(".txt")
            captions: List[str] = []

            if not txt_path.exists():
                # Search the same class directory for a filename sharing the stem.
                cand = list((captions_text_root / rel.parent).glob(rel.stem + "*.txt"))
                if len(cand) == 1:
                    txt_path = cand[0]
                elif len(cand) > 1:
                    # Prefer the shortest matching filename.
                    cand.sort(key=lambda p: len(p.name))
                    txt_path = cand[0]

            if txt_path.exists():
                try:
                    with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
                        lines = [ln.strip() for ln in f.readlines()]
                        captions = [ln for ln in lines if len(ln) > 0]
                except Exception as e:
                    print(f"Warning: Unable to read {txt_path}: {e}")
            else:
                miss += 1

            if len(captions) == 0:
                pass

            # Keep raw relative paths as keys until split-specific export.
            idx[str(rel)] = captions

        if miss > 0:
            print(f"Warning: Caption files missing for {miss}/{total} images; storing empty lists")
        else:
            print("Caption files found for all images")

        self._captions_index = idx
        return idx

    def _save_coco_captions_json(self, metadata: Dict[str, pd.DataFrame]):
        """
        Write COCO-format caption JSON files for train2017 and val2017.

        Image filenames are relative to each split directory, including the class
        subdirectory. Each caption becomes a separate annotation.
        """
        assert self._metadata_df is not None
        assert self._classes_df is not None
        assert self.coco_compat, "COCO JSON export requires coco_compat=True"

        meta = self._metadata_df
        classes = self._classes_df

        ann_dir = self.processed_dir / "annotations"
        ann_dir.mkdir(parents=True, exist_ok=True)

        splits = {
            "train2017": meta[meta["is_training_img"] == 1],
            "val2017": meta[meta["is_training_img"] == 0],
        }

        categories = []
        for r in classes.itertuples(index=False):
            categories.append({
                "id": int(r.class_id),
                "name": str(r.class_name),
                "supercategory": "bird"
            })

        for split_name, df_split in splits.items():
            images, annotations = [], []
            ann_id = 1
            for r in tqdm(df_split.itertuples(index=False), total=len(df_split), desc=f"Building {split_name} JSON"):
                rel_raw = str(Path(r.filepath))  # Relative to raw_dir/images.
                # Retain the class subdirectory under each COCO split.
                rel_for_split = str(Path(r.class_name) / Path(r.filepath).name)

                images.append({
                    "id": int(r.img_id),
                    "file_name": rel_for_split
                })

                caps = self._captions_index.get(rel_raw, [])
                if len(caps) == 0:
                    # Supply one empty caption so the loader can sample a caption.
                    caps = [""]

                for cap in caps:
                    annotations.append({
                        "id": ann_id,
                        "image_id": int(r.img_id),
                        "caption": cap
                    })
                    ann_id += 1

            coco_obj = {
                "info": {
                    "description": "CUB-200-2011 with captions (COCO-style)",
                    "version": "1.0",
                    "year": 2011,
                    "contributor": "CUB community",
                    "date_created": ""
                },
                "licenses": [],
                "images": images,
                "annotations": annotations,
                "categories": categories
            }

            out_json = ann_dir / f"captions_{split_name}.json"
            with open(out_json, "w", encoding="utf-8") as f:
                json.dump(coco_obj, f, ensure_ascii=False)
            print(f"Saved {split_name} annotations: {out_json}  "
                  f"(images={len(images)}, annotations={len(annotations)})")

    # Class metadata and optional source cleanup.

    def save_class_mapping(self, metadata: Dict[str, pd.DataFrame]):
        """Save zero-based class mappings and an ordered class-name list."""
        classes = metadata["classes"]
        class_mapping = {}
        for idx, row in enumerate(classes.itertuples(index=False)):
            class_mapping[idx] = {
                "class_name": row.class_name,
                "original_id": int(row.class_id)
            }
        mapping_file = self.processed_dir / "class_mapping.json"
        with open(mapping_file, "w", encoding="utf-8") as f:
            json.dump(class_mapping, f, indent=2, ensure_ascii=False)

        names_file = self.processed_dir / "class_names.txt"
        with open(names_file, "w", encoding="utf-8") as f:
            f.write("\n".join([r.class_name for r in classes.itertuples(index=False)]))

        print("Class metadata saved:")
        print(f"  Mapping: {mapping_file}")
        print(f"  Class names: {names_file}")

    def cleanup_raw_data(self):
        """Ask before deleting the raw image directory and its downloaded archive."""
        resp = input("Delete the raw image directory and downloaded image archive? (y/N): ")
        if resp.lower() == "y":
            if self.raw_dir.exists():
                shutil.rmtree(self.raw_dir)
                print("Raw image directory deleted")
            dataset_file = self.data_root / self.dataset_filename
            if dataset_file.exists():
                dataset_file.unlink(missing_ok=True)
                print("Image archive deleted")
        else:
            print("Keeping the raw images and archive")

    # End-to-end preparation.

    def process_all(self) -> bool:
        """Download images and captions, organize splits, and export annotations."""
        print("Preparing CUB-200 images and captions...")

        if not self.download_dataset():
            return False
        if not self.extract_dataset():
            return False

        try:
            metadata = self.parse_metadata()
        except Exception as e:
            print(f"Metadata parsing failed: {e}")
            return False

        self.create_imagefolder_structure(metadata)
        self.save_class_mapping(metadata)

        # Reuse manually extracted captions before attempting a download.
        manual_captions_dir = self.data_root / f"cub_captions_{self.captions_source}"
        if manual_captions_dir.exists():
            print(f"Found local caption directory: {manual_captions_dir}")
            candidate_dirs = []
            for p in manual_captions_dir.rglob("*"):
                if p.is_dir() and p.name.lower() in {"text_c10", "text"}:
                    candidate_dirs.append(p)
            
            if candidate_dirs:
                captions_text_root = sorted(candidate_dirs, key=lambda x: x.name.lower() != "text_c10")[0]
                print(f"Using local caption text: {captions_text_root}")
            else:
                print("No text_c10 or text directory found in the local caption directory.")
                return False
        else:
            print("No local caption directory found; downloading captions...")
            captions_zip = self.download_captions()
            if captions_zip is None:
                print("Caption download failed; stopping.")
                return False
            captions_text_root = self.extract_captions(captions_zip)
            if captions_text_root is None:
                print("Caption archive could not be parsed; stopping.")
                return False

        self._load_captions_index(captions_text_root)

        if self.coco_compat:
            self._save_coco_captions_json(metadata)

        print("\nPreprocessing complete")
        print(f"Processed data: {self.processed_dir}")
        if self.coco_compat:
            print("Created train2017/, val2017/, and annotations/captions_*.json")
            print("The prepared data can be loaded with COCODataset.")
        return True


# ----------------------------
# CLI
# ----------------------------

def main():
    parser = argparse.ArgumentParser(description="Prepare CUB-200-2011 images and captions")
    parser.add_argument("--data_root", type=str, required=True, help="Root directory for downloaded and processed data")
    parser.add_argument("--no_coco_compat", action="store_true", help="Skip COCO-style directories and caption JSON export")
    parser.add_argument("--captions_source", type=str, default="reed_cvpr16",
                        choices=["reed_cvpr16", "attngan_preproc"],
                        help="Caption archive to use")
    args = parser.parse_args()

    pre = CUB200Preprocessor(
        data_root=args.data_root,
        coco_compat=not args.no_coco_compat,
        captions_source=args.captions_source
    )

    ok = pre.process_all()
    if ok:
        print("Preprocessing succeeded")
        return 0
    else:
        print("Preprocessing failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
