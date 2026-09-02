import argparse
import os
import random
from typing import List, Tuple

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader

from .dataset_nuinsseg import discover_pairs, NuInsSegDataset, save_pairs_manifest
from .model_unet import UNet
from .transforms import build_transforms
from .utils import (
    set_seed,
    bce_dice_loss,
    dice_coeff,
    iou_score,
    AverageMeter,
    ensure_dir,
    save_checkpoint,
    save_config,
    save_pred_sample,
    count_parameters,
)


def parse_args():
    p = argparse.ArgumentParser(description="Train UNet on NuInsSeg")
    p.add_argument('--data-root', type=str, default='NuInsSeg', help='Root dataset directory')
    p.add_argument('--include', type=str, default='', help='Comma-separated subset filters (partial match)')
    p.add_argument('--outdir', type=str, default='runs/exp1', help='Output directory')
    p.add_argument('--img-size', type=int, default=512, help='Resize square side')
    p.add_argument('--val-split', type=float, default=0.1, help='Validation fraction')
    p.add_argument('--test-split', type=float, default=0.1, help='Test fraction (evaluated at end)')
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--num-workers', type=int, default=4)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--augment', action='store_true', help='Enable data augmentations')
    p.add_argument('--amp', action='store_true', help='Enable mixed precision')
    p.add_argument('--resume', type=str, default='', help='Path to checkpoint to resume')
    p.add_argument('--limit', type=int, default=0, help='Limit number of pairs for debug')
    p.add_argument('--dry-run', action='store_true', help='Only discover pairs and exit')
    p.add_argument('--export-test', action='store_true', help='Export predictions for the test set after evaluation')
    p.add_argument('--export-test-only', action='store_true', help='Skip training; evaluate and export test predictions using best or --resume')
    p.add_argument('--export-dir', type=str, default='test_preds', help='Subdirectory under outdir for test predictions')
    p.add_argument('--pred-threshold', type=float, default=0.5, help='Threshold for binarizing saved predictions')
    return p.parse_args()


def split_pairs_threeway(pairs: List[Tuple[str, str]], val_frac: float, test_frac: float, seed: int):
    if val_frac + test_frac >= 1.0:
        raise ValueError("val-split + test-split must be < 1.0")
    n = len(pairs)
    rng = random.Random(seed)
    idxs = list(range(n))
    rng.shuffle(idxs)

    n_test = int(n * test_frac)
    n_val = int(n * val_frac)
    # ensure at least 1 if nonzero fractions and enough samples
    if test_frac > 0 and n_test == 0 and n >= 1:
        n_test = 1
    if val_frac > 0 and n_val == 0 and n - n_test >= 1:
        n_val = 1

    test_idx = set(idxs[:n_test])
    val_idx = set(idxs[n_test:n_test + n_val])
    train = [pairs[i] for i in idxs if i not in test_idx and i not in val_idx]
    val = [pairs[i] for i in val_idx]
    test = [pairs[i] for i in test_idx]
    return train, val, test


def collate_batch(batch):
    imgs = torch.stack([b.image for b in batch], dim=0)
    msks = torch.stack([b.mask for b in batch], dim=0)
    paths = [(b.img_path, b.msk_path) for b in batch]
    return imgs, msks, paths


def main():
    args = parse_args()
    set_seed(args.seed)

    include = [s.strip() for s in args.include.split(',') if s.strip()] or None
    pairs = discover_pairs(args.data_root, include=include)

    if args.limit and args.limit > 0:
        pairs = pairs[: args.limit]

    print(f"Discovered {len(pairs)} image/mask pairs under {args.data_root}")
    if len(pairs) == 0:
        return

    ensure_dir(args.outdir)
    save_pairs_manifest(pairs, os.path.join(args.outdir, 'pairs.json'))

    if args.dry_run:
        print(f"Manifest saved to {os.path.join(args.outdir, 'pairs.json')}")
        return

    train_pairs, val_pairs, test_pairs = split_pairs_threeway(pairs, args.val_split, args.test_split, args.seed)
    print(f"Train pairs: {len(train_pairs)} | Val pairs: {len(val_pairs)} | Test pairs: {len(test_pairs)}")

    train_t, val_t = build_transforms(args.img_size, args.augment)
    train_ds = NuInsSegDataset(train_pairs, transform=train_t, img_size=args.img_size)
    val_ds = NuInsSegDataset(val_pairs, transform=val_t, img_size=args.img_size)
    test_ds = NuInsSegDataset(test_pairs, transform=val_t, img_size=args.img_size)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, collate_fn=collate_batch
    )
    val_loader = DataLoader(
        val_ds, batch_size=max(1, args.batch_size // 2), shuffle=False,
        num_workers=args.num_workers, pin_memory=True, collate_fn=collate_batch
    )
    test_loader = DataLoader(
        test_ds, batch_size=max(1, args.batch_size // 2), shuffle=False,
        num_workers=args.num_workers, pin_memory=True, collate_fn=collate_batch
    )

    device = torch.device(args.device)
    model = UNet(n_channels=3, n_classes=1)
    model.to(device)
    print(f"Model params: {count_parameters(model)/1e6:.2f}M")

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)
    best_dice = 0.0
    start_epoch = 0

    if args.resume and os.path.isfile(args.resume):
        try:
            ckpt = torch.load(args.resume, map_location='cpu', weights_only=True)
        except TypeError:
            ckpt = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(ckpt['model'])
        opt.load_state_dict(ckpt['opt'])
        if ckpt.get('scaler') is not None:
            try:
                scaler.load_state_dict(ckpt['scaler'])
            except Exception:
                pass
        start_epoch = ckpt.get('epoch', 0) + 1
        best_dice = ckpt.get('best_dice', 0.0)
        print(f"Resumed from {args.resume}: epoch {start_epoch}, best_dice {best_dice:.4f}")

    cfg = vars(args)
    save_config(cfg, args.outdir)

    # Export-only mode: evaluate + export test predictions using existing checkpoint
    if args.export_test_only:
        best_ckpt_path = args.resume if args.resume else os.path.join(args.outdir, 'best.pt')
        if not os.path.isfile(best_ckpt_path):
            print(f"No checkpoint found at {best_ckpt_path}. Provide --resume or ensure best.pt exists in outdir.")
            return
        try:
            ckpt = torch.load(best_ckpt_path, map_location='cpu', weights_only=True)
        except TypeError:
            ckpt = torch.load(best_ckpt_path, map_location='cpu')
        model.load_state_dict(ckpt['model'])
        model.eval()
        test_loss = AverageMeter(); test_dice = AverageMeter(); test_iou = AverageMeter()
        with torch.no_grad():
            for imgs, msks, _ in test_loader:
                imgs = imgs.to(device, non_blocking=True)
                msks = msks.to(device, non_blocking=True)
                logits = model(imgs)
                loss = bce_dice_loss(logits, msks)
                test_loss.update(loss.item(), imgs.size(0))
                test_dice.update(dice_coeff(logits, msks).item(), imgs.size(0))
                test_iou.update(iou_score(logits, msks).item(), imgs.size(0))
        print(f"Test loss {test_loss.avg:.4f} | Dice {test_dice.avg:.4f} | IoU {test_iou.avg:.4f}")

        if args.export_test:
            export_root = os.path.join(args.outdir, args.export_dir)
            ensure_dir(export_root)
            print(f"Exporting test predictions to {export_root} ...")
            with torch.no_grad():
                for imgs, msks, paths in test_loader:
                    imgs = imgs.to(device, non_blocking=True)
                    logits = model(imgs)
                    probs = torch.sigmoid(logits).cpu()
                    for j, (img_path, _) in enumerate(paths):
                        base = os.path.splitext(os.path.basename(img_path))[0]
                        p = probs[j, 0].numpy()
                        prob_img = Image.fromarray((p * 255).astype(np.uint8))
                        bin_img = Image.fromarray(((p > args.pred_threshold).astype(np.uint8) * 255))
                        prob_img.save(os.path.join(export_root, f"{base}_prob.png"))
                        bin_img.save(os.path.join(export_root, f"{base}_pred.png"))
                        save_pred_sample(imgs[j].cpu(), logits[j].cpu(), msks[j].cpu(), os.path.join(export_root, f"{base}_triplet.png"))
            print("Export complete.")
        return

    for epoch in range(start_epoch, args.epochs):
        model.train()
        loss_meter = AverageMeter()
        for step, (imgs, msks, _) in enumerate(train_loader, 1):
            imgs = imgs.to(device, non_blocking=True)
            msks = msks.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=args.amp):
                logits = model(imgs)
                loss = bce_dice_loss(logits, msks)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            loss_meter.update(loss.item(), imgs.size(0))
            if step % 20 == 0:
                print(f"Epoch {epoch+1} | Step {step}/{len(train_loader)} | Loss {loss_meter.avg:.4f}")

        # Validation
        model.eval()
        val_loss = AverageMeter()
        dice_m = AverageMeter()
        iou_m = AverageMeter()
        with torch.no_grad():
            for imgs, msks, paths in val_loader:
                imgs = imgs.to(device, non_blocking=True)
                msks = msks.to(device, non_blocking=True)
                logits = model(imgs)
                loss = bce_dice_loss(logits, msks)
                val_loss.update(loss.item(), imgs.size(0))
                dice_m.update(dice_coeff(logits, msks).item(), imgs.size(0))
                iou_m.update(iou_score(logits, msks).item(), imgs.size(0))

            # Save a small sample grid
            sample_dir = os.path.join(args.outdir, 'samples')
            ensure_dir(sample_dir)
            if len(val_ds) > 0:
                s = val_ds[0]
                s_img = s.image.unsqueeze(0).to(device)
                s_pred = model(s_img)
                save_pred_sample(s.image, s_pred.squeeze(0), s.mask, os.path.join(sample_dir, f'epoch_{epoch+1:03d}.png'))

        print(
            f"Epoch {epoch+1}/{args.epochs} | Train loss {loss_meter.avg:.4f} | "
            f"Val loss {val_loss.avg:.4f} | Dice {dice_m.avg:.4f} | IoU {iou_m.avg:.4f}"
        )

        # Checkpointing
        state = {
            'epoch': epoch,
            'model': model.state_dict(),
            'opt': opt.state_dict(),
            'scaler': scaler.state_dict() if args.amp else None,
            'best_dice': best_dice,
            'config': cfg,
        }
        save_checkpoint(state, args.outdir, 'last.pt')
        if dice_m.avg > best_dice:
            best_dice = dice_m.avg
            state['best_dice'] = best_dice
            save_checkpoint(state, args.outdir, 'best.pt')
            print(f"Saved new best with Dice {best_dice:.4f}")

    # Final test evaluation using best checkpoint
    best_ckpt_path = os.path.join(args.outdir, 'best.pt')
    if os.path.isfile(best_ckpt_path) and len(test_ds) > 0:
        print("Evaluating on test set using best checkpoint...")
        try:
            ckpt = torch.load(best_ckpt_path, map_location='cpu', weights_only=True)
        except TypeError:
            ckpt = torch.load(best_ckpt_path, map_location='cpu')
        model.load_state_dict(ckpt['model'])
        model.eval()
        from .utils import iou_score as _iou, dice_coeff as _dice, AverageMeter as _AM
        test_loss = _AM(); test_dice = _AM(); test_iou = _AM()
        with torch.no_grad():
            for imgs, msks, _ in test_loader:
                imgs = imgs.to(device, non_blocking=True)
                msks = msks.to(device, non_blocking=True)
                logits = model(imgs)
                loss = bce_dice_loss(logits, msks)
                test_loss.update(loss.item(), imgs.size(0))
                test_dice.update(_dice(logits, msks).item(), imgs.size(0))
                test_iou.update(_iou(logits, msks).item(), imgs.size(0))
        print(f"Test loss {test_loss.avg:.4f} | Dice {test_dice.avg:.4f} | IoU {test_iou.avg:.4f}")
        import json
        with open(os.path.join(args.outdir, 'test_metrics.json'), 'w') as f:
            json.dump({
                'loss': test_loss.avg,
                'dice': test_dice.avg,
                'iou': test_iou.avg,
                'num_pairs': len(test_ds),
            }, f, indent=2)
        if args.export_test:
            export_root = os.path.join(args.outdir, args.export_dir)
            ensure_dir(export_root)
            print(f"Exporting test predictions to {export_root} ...")
            with torch.no_grad():
                for imgs, msks, paths in test_loader:
                    imgs = imgs.to(device, non_blocking=True)
                    logits = model(imgs)
                    probs = torch.sigmoid(logits).cpu()
                    for j, (img_path, _) in enumerate(paths):
                        base = os.path.splitext(os.path.basename(img_path))[0]
                        p = probs[j, 0].numpy()
                        prob_img = Image.fromarray((p * 255).astype(np.uint8))
                        bin_img = Image.fromarray(((p > args.pred_threshold).astype(np.uint8) * 255))
                        prob_img.save(os.path.join(export_root, f"{base}_prob.png"))
                        bin_img.save(os.path.join(export_root, f"{base}_pred.png"))
                        save_pred_sample(imgs[j].cpu(), logits[j].cpu(), msks[j].cpu(), os.path.join(export_root, f"{base}_triplet.png"))
            print("Export complete.")


if __name__ == '__main__':
    main()

# Dummy edit marker: 2026-09-02
