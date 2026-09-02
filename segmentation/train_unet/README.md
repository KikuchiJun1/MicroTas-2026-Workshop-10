UNet training for NuInsSeg

This directory contains a from-scratch PyTorch UNet trainer that uses the dataset under `NuInsSeg/` in this repo. It recursively discovers subsets (e.g., `human bladder`, `human brain`, etc.), and pairs images in `tissue images/` with masks in `mask binary/` by matching basenames regardless of extension.

Highlights
- Robust file discovery across all subsets under `NuInsSeg/`
- Supports mask formats: PNG, TIF/TIFF (single page)
- Binary segmentation (background=0, foreground=1)
- Train/val split, augmentations, mixed precision, checkpointing
- Metrics: Dice, IoU

Quick start
- Default run with train/val/test (10%/10%):
  `python -m train_unet.train --data-root NuInsSeg --epochs 50 --batch-size 8 --val-split 0.1 --test-split 0.1`

- Common options:
  - `--img-size 512` target square size (resize)
  - `--val-split 0.1` validation split fraction
  - `--test-split 0.1` test split fraction (evaluated at end with best ckpt)
  - `--num-workers 8` dataloader workers
  - `--augment` enable simple spatial and color augmentations
  - `--outdir runs/exp1` outputs (checkpoints, logs, samples)

Assumptions
- Tissue images are PNG (RGB). Masks are binary (PNG/TIF). If masks are not binary, the loader thresholds them at >0.
- Image/mask pairs are matched by basename (e.g., `human_bladder_01.*`).
- If some pairs do not match, they are skipped and logged.

Files
- `dataset_nuinsseg.py`: Dataset that builds pairs across the whole tree.
- `model_unet.py`: Minimal, well-tested UNet implementation.
- `transforms.py`: Joint transforms for image+mask.
- `utils.py`: Metrics, logging, seeding, misc helpers.
- `train.py`: CLI script for training and validation.

Tips
- If your environment supports GPUs, add `--device cuda` (or `--device cuda:0`).
- To freeze the data discovery to specific subsets, use `--include` with comma-separated folder names (partial matches allowed).
- To inspect discovered pairs without training, run: `python -m train_unet.train --dry-run`.
- After training, test metrics are saved to `<outdir>/test_metrics.json` and a `best.pt` checkpoint is used for evaluation.

<!-- Dummy edit marker: 2026-09-02 -->
