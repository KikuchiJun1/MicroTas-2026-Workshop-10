import os
import json
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset


IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}
MASK_EXTS = {".png", ".tif", ".tiff"}


def _norm_key(name: str) -> str:
    base = os.path.splitext(os.path.basename(name))[0]
    base = base.lower()
    base = base.replace(" ", "_").replace("-", "_")
    # common suffixes
    for suf in ["_mask", "-mask", " mask", "_binary", "-binary"]:
        if base.endswith(suf):
            base = base[: -len(suf)]
            break
    return base


def _is_dir(path: str, name: str) -> bool:
    return os.path.isdir(os.path.join(path, name))


def discover_pairs(
    data_root: str,
    include: Optional[List[str]] = None,
    tissue_dir_name: str = "tissue images",
    mask_dir_name: str = "mask binary",
    image_exts: Optional[set] = None,
    mask_exts: Optional[set] = None,
) -> List[Tuple[str, str]]:
    """
    Walks NuInsSeg structure and returns list of (image_path, mask_path) pairs.

    Matching is by normalized basename ignoring extension and common mask suffixes.
    Non-matching files are skipped.
    """
    image_exts = image_exts or IMG_EXTS
    mask_exts = mask_exts or MASK_EXTS

    if not os.path.isdir(data_root):
        raise FileNotFoundError(f"Data root not found: {data_root}")

    pairs: List[Tuple[str, str]] = []
    subsets = [d for d in os.listdir(data_root) if _is_dir(data_root, d)]

    if include:
        inc = [s.lower() for s in include]
        subsets = [d for d in subsets if any(s in d.lower() for s in inc)]

    for subset in sorted(subsets):
        sub_path = os.path.join(data_root, subset)
        tissue_dir = os.path.join(sub_path, tissue_dir_name)
        mask_dir = os.path.join(sub_path, mask_dir_name)
        if not (os.path.isdir(tissue_dir) and os.path.isdir(mask_dir)):
            continue

        imgs: Dict[str, str] = {}
        msks: Dict[str, str] = {}

        for f in os.listdir(tissue_dir):
            ext = os.path.splitext(f)[1].lower()
            if ext in image_exts:
                key = _norm_key(f)
                imgs[key] = os.path.join(tissue_dir, f)

        for f in os.listdir(mask_dir):
            ext = os.path.splitext(f)[1].lower()
            if ext in mask_exts:
                key = _norm_key(f)
                msks[key] = os.path.join(mask_dir, f)

        # direct matches
        keys = sorted(set(imgs.keys()) & set(msks.keys()))
        for k in keys:
            pairs.append((imgs[k], msks[k]))

    return pairs


@dataclass
class NuInsSegSample:
    image: torch.Tensor  # (C,H,W) float32 in [0,1]
    mask: torch.Tensor   # (1,H,W) float32 in {0,1}
    img_path: str
    msk_path: str


class NuInsSegDataset(Dataset):
    def __init__(
        self,
        pairs: List[Tuple[str, str]],
        transform=None,
        img_size: int = 512,
        threshold: float = 0.0,
    ):
        self.pairs = pairs
        self.transform = transform
        self.img_size = img_size
        self.threshold = threshold

    def __len__(self):
        return len(self.pairs)

    def _load_image(self, path: str) -> Image.Image:
        img = Image.open(path)
        if img.mode != "RGB":
            img = img.convert("RGB")
        return img

    def _load_mask(self, path: str) -> Image.Image:
        m = Image.open(path)
        if m.mode not in ("1", "L"):
            m = m.convert("L")
        return m

    def __getitem__(self, idx: int) -> NuInsSegSample:
        img_path, msk_path = self.pairs[idx]
        img = self._load_image(img_path)
        msk = self._load_mask(msk_path)

        if self.transform is not None:
            img, msk = self.transform(img, msk)

        # convert to tensors
        img_np = np.asarray(img, dtype=np.uint8)
        img_t = torch.from_numpy(img_np).float().permute(2, 0, 1) / 255.0  # (C,H,W)

        msk_np = np.asarray(msk, dtype=np.uint8)
        msk_t = torch.from_numpy((msk_np > self.threshold).astype(np.float32))[None, ...]

        return NuInsSegSample(image=img_t, mask=msk_t, img_path=img_path, msk_path=msk_path)


def save_pairs_manifest(pairs: List[Tuple[str, str]], out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump([{"image": i, "mask": m} for i, m in pairs], f, indent=2)

# Dummy edit marker: 2026-09-02
