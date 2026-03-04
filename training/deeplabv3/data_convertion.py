import os
import cv2
import argparse
from pathlib import Path
import numpy as np
import random
from tqdm import tqdm
import shutil

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

# Class mapping
ROAD_CLASSES = {2, 3, 4, 5}  # -> 1
BOUNDARY_CLASSES = {7}       # -> 2 (drawn last / on top)

def yolo_poly_to_pixels(points, img_w, img_h):
    coords = []
    for i in range(0, len(points), 2):
        x = float(points[i]) * img_w
        y = float(points[i+1]) * img_h
        coords.append((int(round(x)), int(round(y))))
    return np.array(coords, dtype=np.int32)

def find_label_for_image(img_path: Path, input_root: Path) -> Path | None:
    # 1) same folder
    cand = img_path.with_suffix(".txt")
    if cand.exists():
        return cand
    # 2) .../images/... -> .../labels/...
    parts = list(img_path.parts)
    for i, p in enumerate(parts):
        if p.lower() == "images":
            parts[i] = "labels"
            cand2 = Path(*parts).with_suffix(".txt")
            if cand2.exists():
                return cand2
    # 3) sibling labels next to images/depth: <scene>/labels/<name>.txt
    if img_path.parent.parent.exists():
        sibling = img_path.parent.parent / "labels" / (img_path.stem + ".txt")
        if sibling.exists():
            return sibling
    # 4) fallback: <input_root>/labels/<name>.txt
    try:
        rel = img_path.relative_to(input_root)
        cand3 = input_root / "labels" / rel.with_suffix(".txt").name
        if cand3.exists():
            return cand3
    except Exception:
        pass
    return None

def find_depth_for_image(img_path: Path) -> Path | None:
    """Depth expected at <scene>/depth/<same-filename>."""
    scene_dir = img_path.parent.parent
    if not scene_dir.exists():
        return None
    depth_path = scene_dir / "depth" / img_path.name
    return depth_path if depth_path.exists() else None

def draw_mask_from_txt(label_path: Path, h: int, w: int) -> np.ndarray:
    """Background=0, Road=1 (2/3/4/5), Boundary=2 (7 drawn last)."""
    mask = np.zeros((h, w), dtype=np.uint8)
    if label_path is None or not label_path.exists():
        return mask

    road_polys, boundary_polys = [], []
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 7:
                continue
            try:
                cls = int(float(parts[0]))
            except ValueError:
                continue
            coords = yolo_poly_to_pixels(parts[1:], w, h)
            if len(coords) < 3:
                continue
            if cls in ROAD_CLASSES:
                road_polys.append(coords)
            elif cls in BOUNDARY_CLASSES:
                boundary_polys.append(coords)

    if road_polys:
        cv2.fillPoly(mask, road_polys, color=1)
    if boundary_polys:  # draw last, on top
        cv2.fillPoly(mask, boundary_polys, color=2)
    return mask

def resize_to_height_then_center_crop_square(img: np.ndarray, target=512, is_mask=False) -> np.ndarray:
    """Resize to height=target, then center crop to 512x512.
       If width<target after first resize, scale to width=target then crop vertically.
    """
    h, w = img.shape[:2]
    if h == 0 or w == 0:
        return np.zeros((target, target), dtype=img.dtype)

    # Step 1: height -> target
    scale = target / float(h)
    new_w = max(1, int(round(w * scale)))
    interp_img = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    interp = cv2.INTER_NEAREST if is_mask else interp_img
    resized = cv2.resize(img, (new_w, target), interpolation=interp)

    # Step 2: crop or expand to square
    if new_w >= target:
        left = (new_w - target) // 2
        return resized[:, left:left + target]
    else:
        # Make width target, height grows proportionally, then center-crop vertically
        scale2 = target / float(new_w)
        new_h2 = max(1, int(round(target * scale2)))
        interp2 = cv2.INTER_NEAREST if is_mask else (cv2.INTER_AREA if scale2 < 1.0 else cv2.INTER_LINEAR)
        resized2 = cv2.resize(resized, (target, new_h2), interpolation=interp2)
        top = (new_h2 - target) // 2
        return resized2[top:top + target, :]

def make_square_512_pair(img: np.ndarray, mask: np.ndarray, target=512):
    """Apply same policy to image and mask; enforce exact same size at the end."""
    img_  = resize_to_height_then_center_crop_square(img,  target=target, is_mask=False)
    mask_ = resize_to_height_then_center_crop_square(mask, target=target, is_mask=True)
    if img_.shape[:2] != mask_.shape[:2]:
        mask_ = cv2.resize(mask_, (img_.shape[1], img_.shape[0]), interpolation=cv2.INTER_NEAREST)
    return img_, mask_

def clean_dir(path: Path):
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)

def main():
    ap = argparse.ArgumentParser(description="Build 512x512 RGB/Mask/Depth dataset with strict 1:1 saving (mask-first).")
    ap.add_argument("--input_dir", required=True, type=Path, help="Root with scenes containing images/, labels/, and optionally depth/")
    ap.add_argument("--output_dir", required=True, type=Path, help="Destination root")
    ap.add_argument("--val_ratio", type=float, default=0.2, help="Validation ratio (default 0.2)")
    ap.add_argument("--seed", type=int, default=42, help="Random seed for splitting")
    ap.add_argument("--clean", action="store_true", help="Delete existing output train/ and val/ before writing")
    args = ap.parse_args()

    input_root: Path = args.input_dir.resolve()
    output_root: Path = args.output_dir.resolve()
    random.seed(args.seed)

    # --- Collect ONLY RGB images under an 'images' directory; skip any path that contains 'depth'
    all_rgb_images = []
    for dirpath, _, filenames in os.walk(input_root):
        dirpath_path = Path(dirpath)
        parts_lower = [p.lower() for p in dirpath_path.parts]
        if "depth" in parts_lower:
            continue
        if "images" not in parts_lower:
            continue
        for fname in filenames:
            if Path(fname).suffix.lower() in IMG_EXTS:
                all_rgb_images.append(dirpath_path / fname)

    # Filter to those that have a label
    rgb_with_labels = []
    for img_path in all_rgb_images:
        lbl = find_label_for_image(img_path, input_root)
        if lbl and lbl.exists():
            rgb_with_labels.append(img_path)

    dataset = []
    for img_path in tqdm(rgb_with_labels, desc="Preparing samples", unit="img"):
        rgb = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if rgb is None:
            continue
        h, w = rgb.shape[:2]
        label_path = find_label_for_image(img_path, input_root)
        mask = draw_mask_from_txt(label_path, h, w)
        rgb_512, mask_512 = make_square_512_pair(rgb, mask, target=512)
        depth_512 = None
        depth_path = find_depth_for_image(img_path)
        if depth_path is not None:
            depth_img = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
            if depth_img is not None:
                depth_512 = resize_to_height_then_center_crop_square(depth_img, target=512, is_mask=False)

        unique_name = f"{img_path.parent.name}_{img_path.name}"
        dataset.append((unique_name, rgb_512, mask_512, depth_512))

    # Split
    random.shuffle(dataset)
    val_count = int(len(dataset) * args.val_ratio)
    val_set = dataset[:val_count]
    train_set = dataset[val_count:]

    # Output structure
    train_img_dir = output_root / "train" / "images"
    train_lbl_dir = output_root / "train" / "labels"
    train_dep_dir = output_root / "train" / "depth"
    val_img_dir   = output_root / "val" / "images"
    val_lbl_dir   = output_root / "val" / "labels"
    val_dep_dir   = output_root / "val" / "depth"

    if args.clean:
        for d in [train_img_dir, train_lbl_dir, train_dep_dir, val_img_dir, val_lbl_dir, val_dep_dir]:
            clean_dir(d)
    else:
        for d in [train_img_dir, train_lbl_dir, train_dep_dir, val_img_dir, val_lbl_dir, val_dep_dir]:
            d.mkdir(parents=True, exist_ok=True)

    # Save (MASK-FIRST): only write RGB and depth if mask write succeeds
    stats = {
        "train": {"saved": 0, "mask_fail": 0, "img_fail": 0},
        "val":   {"saved": 0, "mask_fail": 0, "img_fail": 0},
    }

    for subset, img_dir, lbl_dir, dep_dir, name, key in [
        (train_set, train_img_dir, train_lbl_dir, train_dep_dir, "Saving train set", "train"),
        (val_set,   val_img_dir,   val_lbl_dir,   val_dep_dir,   "Saving val set",   "val"),
    ]:
        for fname, rgb_512, mask_512, depth_512 in tqdm(subset, desc=name, unit="file"):
            mask_path = lbl_dir / (Path(fname).stem + ".png")
            if not cv2.imwrite(str(mask_path), mask_512):
                stats[key]["mask_fail"] += 1
                # Do NOT write RGB/depth if mask failed
                continue

            img_path_out = img_dir / fname
            if not cv2.imwrite(str(img_path_out), rgb_512):
                stats[key]["img_fail"] += 1
                # If RGB failed, remove the mask we just wrote to keep 1:1
                try:
                    mask_path.unlink(missing_ok=True)
                except TypeError:
                    # Python <3.8 compatibility
                    if mask_path.exists():
                        mask_path.unlink()
                continue

            # Depth is optional: if available, write it now (doesn't affect RGB/mask pairing)
            if depth_512 is not None:
                cv2.imwrite(str(dep_dir / fname), depth_512)

            stats[key]["saved"] += 1

    print("\n✅ Done.")
    print(f"Train samples saved: {stats['train']['saved']} (mask_fail={stats['train']['mask_fail']}, img_fail={stats['train']['img_fail']})")
    print(f"Val   samples saved: {stats['val']['saved']} (mask_fail={stats['val']['mask_fail']}, img_fail={stats['val']['img_fail']})")
    print(f"Output at: {output_root}")

if __name__ == "__main__":
    main()
