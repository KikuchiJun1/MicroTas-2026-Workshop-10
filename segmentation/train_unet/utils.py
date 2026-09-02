import os
import json
import random
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def count_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def dice_coeff(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    pred = torch.sigmoid(pred)
    pred = (pred > 0.5).float()
    target = (target > 0.5).float()
    inter = (pred * target).sum(dim=(1, 2, 3))
    union = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    dice = (2 * inter + eps) / (union + eps)
    return dice.mean()


def iou_score(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    pred = torch.sigmoid(pred)
    pred = (pred > 0.5).float()
    target = (target > 0.5).float()
    inter = (pred * target).sum(dim=(1, 2, 3))
    union = (pred + target - pred * target).sum(dim=(1, 2, 3))
    iou = (inter + eps) / (union + eps)
    return iou.mean()


def bce_dice_loss(logits: torch.Tensor, targets: torch.Tensor, bce_weight: float = 0.5) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, targets)
    probs = torch.sigmoid(logits)
    inter = (probs * targets).sum(dim=(1, 2, 3))
    union = probs.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
    dice = (2 * inter + 1e-6) / (union + 1e-6)
    dice_loss = 1 - dice.mean()
    return bce_weight * bce + (1 - bce_weight) * dice_loss


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / max(1, self.count)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def save_checkpoint(state: dict, outdir: str, filename: str = "checkpoint.pt"):
    ensure_dir(outdir)
    path = os.path.join(outdir, filename)
    torch.save(state, path)
    return path


def save_config(cfg: dict, outdir: str):
    ensure_dir(outdir)
    with open(os.path.join(outdir, 'config.json'), 'w') as f:
        json.dump(cfg, f, indent=2)


def save_pred_sample(image: torch.Tensor, pred: torch.Tensor, target: torch.Tensor, out_path: str):
    # image: (C,H,W) in [0,1], pred: logits (1,H,W), target: (1,H,W)
    img = (image.clamp(0, 1) * 255).byte().cpu().permute(1, 2, 0).numpy()
    p = torch.sigmoid(pred).cpu().squeeze(0).numpy()
    t = target.cpu().squeeze(0).numpy()
    p_img = (p > 0.5).astype(np.uint8) * 255
    t_img = (t > 0.5).astype(np.uint8) * 255
    # Side-by-side composite
    h, w = img.shape[:2]
    canvas = Image.new('RGB', (w * 3, h))
    canvas.paste(Image.fromarray(img), (0, 0))
    canvas.paste(Image.fromarray(np.stack([p_img]*3, axis=-1)), (w, 0))
    canvas.paste(Image.fromarray(np.stack([t_img]*3, axis=-1)), (2*w, 0))
    canvas.save(out_path)


# Dummy edit marker: 2026-09-02
