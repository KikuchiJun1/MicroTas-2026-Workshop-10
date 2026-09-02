import random
from typing import Tuple, Callable

from PIL import Image, ImageOps
try:
    from torchvision.transforms import ColorJitter
except Exception:
    ColorJitter = None  # optional


def joint_resize(img: Image.Image, msk: Image.Image, size: int) -> Tuple[Image.Image, Image.Image]:
    img = img.resize((size, size), Image.BILINEAR)
    msk = msk.resize((size, size), Image.NEAREST)
    return img, msk


def joint_hflip(img: Image.Image, msk: Image.Image, p: float = 0.5):
    if random.random() < p:
        img = ImageOps.mirror(img)
        msk = ImageOps.mirror(msk)
    return img, msk


def joint_vflip(img: Image.Image, msk: Image.Image, p: float = 0.5):
    if random.random() < p:
        img = ImageOps.flip(img)
        msk = ImageOps.flip(msk)
    return img, msk


def joint_rotate90(img: Image.Image, msk: Image.Image, p: float = 0.5):
    if random.random() < p:
        k = random.choice([1, 2, 3])
        img = img.rotate(90 * k, expand=True)
        msk = msk.rotate(90 * k, expand=True)
    return img, msk


def build_transforms(img_size: int, augment: bool) -> Tuple[Callable, Callable]:
    def train_t(img: Image.Image, msk: Image.Image):
        img, msk = joint_resize(img, msk, img_size)
        if augment:
            img, msk = joint_hflip(img, msk, 0.5)
            img, msk = joint_vflip(img, msk, 0.5)
            img, msk = joint_rotate90(img, msk, 0.5)
            if ColorJitter is not None:
                jitter = ColorJitter(0.1, 0.1, 0.1, 0.05)
                img = jitter(img)
        return img, msk

    def val_t(img: Image.Image, msk: Image.Image):
        img, msk = joint_resize(img, msk, img_size)
        return img, msk

    return train_t, val_t


# Dummy edit marker: 2026-09-02
