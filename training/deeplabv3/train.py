# train_forest_seg.py
# Python 3.10+
# Minimal deps: pip install torch torchvision segmentation-models-pytorch albumentations opencv-python tqdm
# Optional but recommended: pip install timm  (for many SMP encoders)

import os, argparse, time, math, random
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp
import albumentations as A

# -----------------------
# Utils
# -----------------------
def set_seed(seed: int = 42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def ensure_dir(p):
    Path(p).mkdir(parents=True, exist_ok=True)

def colorize_mask(mask, palette=None):
    # quick viz palette for 4 classes
    if palette is None:
        palette = {
            0: (0,0,0),        # bg
            1: (70, 130, 180), # road (steel blue)
            2: (220, 20, 60),  # left boundary (crimson)
            3: (60, 179, 113), # right boundary (medium sea green)
        }
    h, w = mask.shape
    vis = np.zeros((h,w,3), dtype=np.uint8)
    for k, c in palette.items():
        vis[mask==k] = c
    return vis

# -----------------------
# Sample overlay helpers
# -----------------------
def pick_sample_names(val_img_dir, k=5):
    names = sorted([n for n in os.listdir(val_img_dir) if n.lower().endswith(('.jpg','.jpeg','.png'))])
    if len(names) <= k:
        return names
    return random.sample(names, k)

def overlay_mask(rgb, mask, alpha=0.5, palette=None):
    """rgb: HxWx3 uint8 (RGB), mask: HxW uint8 labels"""
    color = colorize_mask(mask, palette)  # RGB
    overlay = (alpha * color + (1 - alpha) * rgb).astype(np.uint8)
    return overlay

def build_val_transform(img_size):
    h, w = img_size
    return A.Compose([
        A.LongestMaxSize(max_size=max(h,w)),
        A.PadIfNeeded(min_height=h, min_width=w, border_mode=cv2.BORDER_CONSTANT, value=(0,0,0)),
        A.CenterCrop(height=h, width=w),
    ])

# -----------------------
# Dataset (RGBD optional per-sample)
# -----------------------
class RoadDataset(Dataset):
    """
    Expects:
      images/: img_0001.jpg, ...
      masks/:  img_0001.png  (single-channel PNG with values {0,1,2,3})
    Optional depth as .npy or 16-bit PNG with identical stem and directory:
      depth/:  img_0001.npy  or img_0001.png

    If use_depth=True but a depth file is missing, a zero depth map is used so
    batches always have 4 channels (RGB + depth).
    """
    def __init__(self, img_dir, mask_dir, depth_dir=None, use_depth=False, img_size=(512,512), augment=True):
        self.img_dir = Path(img_dir)
        self.mask_dir = Path(mask_dir)
        self.depth_dir = Path(depth_dir) if depth_dir else None
        self.use_depth = use_depth
        self.img_size = img_size
        self.augment = augment

        self.names = sorted([n for n in os.listdir(self.img_dir) if n.lower().endswith(('.jpg','.jpeg','.png'))])
        assert len(self.names)>0, f"No images found in {img_dir}"

        # Albumentations pipeline
        h, w = img_size
        if augment:
            self.tf = A.Compose([
                A.LongestMaxSize(max_size=max(h,w)),
                A.PadIfNeeded(min_height=h, min_width=w, border_mode=cv2.BORDER_CONSTANT, value=(0,0,0)),
                A.RandomCrop(height=h, width=w),
                A.HorizontalFlip(p=0.5),
                A.RandomRotate90(p=0.2),
                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05, p=0.5),
                A.GaussNoise(var_limit=(5.0, 20.0), p=0.2),
            ])
        else:
            self.tf = A.Compose([
                A.LongestMaxSize(max_size=max(h,w)),
                A.PadIfNeeded(min_height=h, min_width=w, border_mode=cv2.BORDER_CONSTANT, value=(0,0,0)),
                A.CenterCrop(height=h, width=w),
            ])

    def __len__(self): return len(self.names)

    def _read_depth(self, stem):
        if not self.depth_dir: return None
        # Try .npy first
        npy = self.depth_dir / f"{stem}.npy"
        if npy.exists():
            d = np.load(str(npy))
            return d
        # Then 16-bit PNG
        png = self.depth_dir / f"{stem}.png"
        if png.exists():
            d16 = cv2.imread(str(png), cv2.IMREAD_UNCHANGED)
            if d16 is None: return None
            return d16.astype(np.float32)
        return None

    def __getitem__(self, i):
        name = self.names[i]
        stem = Path(name).stem

        img = cv2.imread(str(self.img_dir / name))
        if img is None: raise RuntimeError(f"Failed to read {name}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(str(self.mask_dir / f"{stem}.png"), cv2.IMREAD_GRAYSCALE)
        if mask is None: raise RuntimeError(f"Missing mask for {name}")

        # Prepare depth (may be None); if use_depth=True and missing, make zeros NOW
        if self.use_depth:
            d = self._read_depth(stem)
            if d is None:
                d = np.zeros(img.shape[:2], dtype=np.float32)
                had_real_depth = False
            else:
                d = d.astype(np.float32)
                # robust normalisation AFTER transforms is also fine; we clip here first
                d = np.clip(d, 0, np.percentile(d, 99.5))
                if d.max() > 1e-6: d = d / (d.max()+1e-6)
                had_real_depth = True
        else:
            d = None
            had_real_depth = False

        # ---- Albumentations: depth MUST be treated as 'mask' to avoid pixel-noise ops ----
        if self.use_depth:
            add_targets = {'mask':'mask', 'depth':'mask'}  # <<-- key fix
            data = {'image': img, 'mask': mask, 'depth': d}
        else:
            add_targets = {'mask':'mask'}
            data = {'image': img, 'mask': mask}

        tf = A.Compose(self.tf.transforms, additional_targets=add_targets)
        aug = tf(**data)
        img, mask = aug['image'], aug['mask']
        if self.use_depth:
            d = aug['depth']

        # If we injected zeros earlier (no real depth), d remains zeros and is aligned.

        # --- Normalise RGB ---
        img = img.astype(np.float32) / 255.0
        mean = np.array([0.485,0.456,0.406], dtype=np.float32)
        std  = np.array([0.229,0.224,0.225], dtype=np.float32)
        img = (img - mean) / std

        # --- Stack channels (always 4 if use_depth=True) ---
        if self.use_depth:
            # Re-normalise depth AFTER transforms (in case scaling changed range)
            if had_real_depth:
                d = np.clip(d, 0, np.percentile(d, 99.5))
                if d.max() > 1e-6: d = d / (d.max()+1e-6)
            d = d[..., None]  # HWC 1
            img = np.concatenate([img, d], axis=-1)  # HWC (4)
        # else: keep RGB only (HWC 3)

        x = np.transpose(img, (2,0,1))  # CHW
        mask = mask.astype(np.int64)

        return torch.from_numpy(x), torch.from_numpy(mask)

# -----------------------
# Metrics
# -----------------------
@torch.no_grad()
def compute_iou_per_class(logits, targets, num_classes):
    """
    logits: (N,C,H,W), targets: (N,H,W)
    returns list of IoU per class
    """
    preds = torch.argmax(logits, dim=1)
    ious = []
    for c in range(num_classes):
        pred_c = (preds == c)
        targ_c = (targets == c)
        inter = (pred_c & targ_c).sum().item()
        union = (pred_c | targ_c).sum().item()
        iou = inter/union if union>0 else float('nan')
        ious.append(iou)
    return ious

# -----------------------
# Visualization: save 5 random samples each epoch (GT & PRED + collage)
# -----------------------
@torch.no_grad()
def save_val_samples(model, args, sample_names, val_tf_vis, epoch, device):
    """
    Save per-sample overlays (_gt.png and _pred.png) with fixed names
    (sample_1_gt.png, sample_1_pred.png, etc.) and a collage.png
    that is overwritten every epoch.
    """
    model.eval()
    out_dir = Path(args.save_dir) / "samples"
    ensure_dir(out_dir)

    row_images = []  # for collage

    for idx, name in enumerate(sample_names, start=1):
        stem = Path(name).stem

        # --- Load RGB ---
        img_bgr = cv2.imread(str(Path(args.val_images) / name))
        if img_bgr is None:
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # --- Load ground truth mask ---
        gt_mask = cv2.imread(str(Path(args.val_masks) / f"{stem}.png"), cv2.IMREAD_GRAYSCALE)
        if gt_mask is None:
            continue

        # --- Optional depth (only for consistent spatial tf; not visualized) ---
        if args.use_depth and args.depth_val:
            npy = Path(args.depth_val) / f"{stem}.npy"
            png = Path(args.depth_val) / f"{stem}.png"
            if npy.exists():
                d = np.load(str(npy)).astype(np.float32)
            elif png.exists():
                d16 = cv2.imread(str(png), cv2.IMREAD_UNCHANGED)
                d = d16.astype(np.float32) if d16 is not None else None
            else:
                d = None
            if d is None:
                d = np.zeros(img_rgb.shape[:2], dtype=np.float32)
            else:
                d = np.clip(d, 0, np.percentile(d, 99.5))
                if d.max() > 1e-6: d = d / (d.max()+1e-6)
        else:
            d = None

        # --- Apply same spatial transform (depth as 'mask'!) ---
        if d is None:
            add_targets = {'mask':'mask'}
            data = {'image': img_rgb, 'mask': gt_mask}
        else:
            add_targets = {'mask':'mask', 'depth':'mask'}  # <<-- important
            data = {'image': img_rgb, 'mask': gt_mask, 'depth': d}

        tf = A.Compose(val_tf_vis.transforms, additional_targets=add_targets)
        aug = tf(**data)
        vis_rgb, gt_mask = aug['image'], aug['mask']
        if d is not None:
            d = aug['depth']

        # --- Prepare tensor for prediction ---
        img_norm = vis_rgb.astype(np.float32) / 255.0
        mean = np.array([0.485,0.456,0.406], dtype=np.float32)
        std  = np.array([0.229,0.224,0.225], dtype=np.float32)
        img_norm = (img_norm - mean) / std

        if args.use_depth:
            if d is None:
                d = np.zeros(vis_rgb.shape[:2], dtype=np.float32)
            d = d[..., None]
            inp = np.concatenate([img_norm, d], axis=-1)
        else:
            inp = img_norm

        tensor = torch.from_numpy(np.transpose(inp, (2,0,1))).unsqueeze(0).to(device)

        with torch.cuda.amp.autocast(enabled=True):
            logits = model(tensor)
            pred_mask = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

        # --- Create overlays ---
        overlay_gt   = overlay_mask(vis_rgb, gt_mask, alpha=0.5)
        overlay_pred = overlay_mask(vis_rgb, pred_mask, alpha=0.5)

        # --- Save with fixed names ---
        cv2.imwrite(str(out_dir / f"sample_{idx}_gt.png"),   cv2.cvtColor(overlay_gt, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(out_dir / f"sample_{idx}_pred.png"), cv2.cvtColor(overlay_pred, cv2.COLOR_RGB2BGR))

        # --- For collage ---
        row = np.concatenate([overlay_gt, overlay_pred], axis=1)
        row_images.append(row)

    # --- Save collage (fixed name) ---
    if row_images:
        collage = np.concatenate(row_images, axis=0)
        collage_bgr = cv2.cvtColor(collage, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(out_dir / "collage.png"), collage_bgr)

# -----------------------
# Training
# -----------------------
def train(args):
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ensure_dir(args.save_dir)
    with open(Path(args.save_dir)/'args.txt','w') as f:
        f.write(str(vars(args)))

    # Datasets/DataLoaders
    train_ds = RoadDataset(args.train_images, args.train_masks,
                           depth_dir=args.depth_train, use_depth=args.use_depth,
                           img_size=tuple(args.img_size), augment=True)
    val_ds   = RoadDataset(args.val_images, args.val_masks,
                           depth_dir=args.depth_val, use_depth=args.use_depth,
                           img_size=tuple(args.img_size), augment=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds, batch_size=max(1,args.batch_size//2), shuffle=False,
                              num_workers=args.workers, pin_memory=True)

    samples_dir = Path(args.save_dir) / "samples"
    ensure_dir(samples_dir)
    val_tf_vis = build_val_transform(tuple(args.img_size))

    # Model
    in_ch = 4 if args.use_depth else 3
    model = smp.DeepLabV3Plus(
        encoder_name=args.encoder,
        encoder_weights='imagenet',
        in_channels=in_ch,
        classes=args.classes,
        activation=None
    ).to(device)

    # Loss / Optim
    if args.class_weights is not None:
        w = torch.tensor(args.class_weights, dtype=torch.float32, device=device)
    else:
        # default: emphasise thin boundaries
        w = torch.tensor([0.25, 1.0, 3.0, 3.0][:args.classes], dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=w)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    scaler = torch.cuda.amp.GradScaler(enabled=not args.no_amp)

    best_miou = -1.0
    start = time.time()

    for epoch in range(1, args.epochs+1):
        model.train()
        losses = []
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", ncols=100)
        for imgs, masks in pbar:
            imgs = imgs.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=not args.no_amp):
                logits = model(imgs)
                loss = criterion(logits, masks)
            scaler.scale(loss).backward()
            if args.clip_grad is not None:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            scaler.step(optimizer)
            scaler.update()
            losses.append(loss.item())
            pbar.set_postfix(loss=f"{np.mean(losses):.4f}")

        # Validation
        model.eval()
        val_losses = []
        iou_sums = np.zeros(args.classes, dtype=np.float64)
        iou_counts = np.zeros(args.classes, dtype=np.int64)

        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs = imgs.to(device, non_blocking=True)
                masks = masks.to(device, non_blocking=True)
                with torch.cuda.amp.autocast(enabled=not args.no_amp):
                    logits = model(imgs)
                    vloss = criterion(logits, masks)
                val_losses.append(vloss.item())
                ious = compute_iou_per_class(logits, masks, args.classes)
                for c, iou in enumerate(ious):
                    if not math.isnan(iou):
                        iou_sums[c] += iou
                        iou_counts[c] += 1

        per_class_iou = (iou_sums / np.maximum(iou_counts, 1)).tolist()
        miou = float(np.nanmean(per_class_iou))
        print(f"\nVal: loss={np.mean(val_losses):.4f} | mIoU={miou:.4f} | per-class IoU={['%.3f'%x for x in per_class_iou]}")

        # Save 5 random visual samples (overwrites each epoch)
        sample_names = pick_sample_names(args.val_images, k=5)
        save_val_samples(model, args, sample_names, val_tf_vis, epoch, device)

        # Save last & best
        last_path = Path(args.save_dir) / "last.pt"
        torch.save({
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scaler': scaler.state_dict(),
            'args': vars(args),
            'miou': miou
        }, last_path)

        if miou > best_miou:
            best_miou = miou
            best_path = Path(args.save_dir) / "best.pt"
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scaler': scaler.state_dict(),
                'args': vars(args),
                'miou': miou
            }, best_path)
            print(f"  ✓ Saved new best to {best_path} (mIoU {miou:.4f})")

        # Early stopping by wall-time (optional guard for 24h)
        if args.max_hours is not None and (time.time()-start) > args.max_hours*3600:
            print("Reached max_hours budget, stopping.")
            break

    print(f"Training complete. Best mIoU={best_miou:.4f}")

# -----------------------
# Inference helper (quick sanity check)
# -----------------------
@torch.no_grad()
def demo_one(args, image_path, ckpt_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt = torch.load(ckpt_path, map_location=device)
    in_ch = 4 if args.use_depth else 3
    model = smp.DeepLabV3Plus(
        encoder_name=args.encoder, encoder_weights=None,
        in_channels=in_ch, classes=args.classes, activation=None
    ).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()

    img_bgr = cv2.imread(image_path); img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w = args.img_size
    tf = A.Compose([
        A.LongestMaxSize(max_size=max(h,w)),
        A.PadIfNeeded(min_height=h, min_width=w, border_mode=cv2.BORDER_CONSTANT, value=(0,0,0)),
        A.CenterCrop(height=h, width=w),
    ])
    aug = tf(image=img); img = aug['image'].astype(np.float32)/255.0
    mean = np.array([0.485,0.456,0.406], dtype=np.float32)
    std  = np.array([0.229,0.224,0.225], dtype=np.float32)
    img = (img-mean)/std

    if args.use_depth:
        H, W, _ = img.shape
        depth = np.zeros((H,W,1), dtype=np.float32)  # demo: no depth -> zeros
        img = np.concatenate([img, depth], axis=-1)

    img = np.transpose(img, (2,0,1))[None]  # 1CHW
    img = torch.from_numpy(img).to(device)

    with torch.cuda.amp.autocast(enabled=True):
        logits = model(img)
        pred = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

    vis = colorize_mask(pred)
    out_path = Path(args.save_dir)/"demo_pred.png"
    cv2.imwrite(str(out_path), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
    print(f"Saved demo prediction to {out_path}")

# -----------------------
# CLI
# -----------------------
def parse_args():
    p = argparse.ArgumentParser("Forest road segmentation training (DeepLabV3+)")
    p.add_argument('--train-images', required=True)
    p.add_argument('--train-masks',  required=True)
    p.add_argument('--val-images',   required=True)
    p.add_argument('--val-masks',    required=True)
    p.add_argument('--depth-train',  default=None, help='Depth dir for train (optional)')
    p.add_argument('--depth-val',    default=None, help='Depth dir for val (optional)')
    p.add_argument('--use-depth', action='store_true', help='Use RGBD input (4 channels, zeros when missing)')

    p.add_argument('--encoder', default='timm-mobilenetv3_small_100',
                   help='Any SMP encoder name (e.g., resnet50, timm-mobilenetv3_small_100)')
    p.add_argument('--classes', type=int, default=4)
    p.add_argument('--img-size', nargs=2, type=int, default=[512,512])
    p.add_argument('--batch-size', type=int, default=2)
    p.add_argument('--epochs', type=int, default=60)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--weight-decay', type=float, default=1e-4)
    p.add_argument('--class-weights', nargs='*', type=float, default=None,
                   help='e.g., --class-weights 0.25 1 3 3')
    p.add_argument('--clip-grad', type=float, default=None)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--save-dir', default='runs/exp')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--no-amp', action='store_true', help='Disable mixed precision')
    p.add_argument('--max-hours', type=float, default=None, help='Stop after this many hours (safety for 24h budget)')
    p.add_argument('--demo-image', default=None, help='Optional: run a single-image demo with best checkpoint')
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    ensure_dir(args.save_dir)
    train(args)
    if args.demo_image:
        best_ckpt = Path(args.save_dir)/"best.pt"
        if best_ckpt.exists():
            demo_one(args, args.demo_image, str(best_ckpt))
        else:
            print("No best.pt found for demo.")
