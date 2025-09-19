import os
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

# -----------------------------
# Settings
# -----------------------------
MODEL_PATH = "yolo_V11/runs/segment/UNI_finetuning/weights/best.pt"                # your trained model
IMAGE_PATHS = [
    "baseline/image_01.png",
    "baseline/image_03.jpg",
    "baseline/image_05.png",
    "baseline/image_06.png",
    "baseline/image_07.png",
    "baseline/image_08.png",
    "baseline/image_09.jpeg",
    "baseline/image_10.png",
    "baseline/image_11.png",
    "baseline/image_12.png"
]
OUTPUT_DIR = "baseline/output/baseline_yolo"
ALPHA = 0.5                                     # overlay transparency

# Which classes to draw (by class id from your model)
allowed_class_ids = [7]

# Collage options
GRID_COLS = 3
GRID_BG_COLOR = (255, 255, 255)                 # white background
GRID_FILENAME = "overlay_grid.png"
COLLAGE_TITLE = "YOLO"   # <- your title text
# OPTIONAL fixed tile size (H, W). Comment out to auto-size from largest image.
# FORCE_CELL_SIZE = (720, 1280)

# -----------------------------
# Setup
# -----------------------------
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load YOLOv8 segmentation model
model = YOLO(MODEL_PATH)

# -----------------------------
# Helpers
# -----------------------------
def letterbox_to_size(img: np.ndarray, target_h: int, target_w: int,
                      bg_color=(255, 255, 255)) -> np.ndarray:
    """Resize to fit (target_h, target_w) preserving aspect ratio, pad with bg_color."""
    h, w = img.shape[:2]
    scale = min(target_w / w, target_h / h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(img, (new_w, new_h), interpolation=interp)
    canvas = np.full((target_h, target_w, 3), bg_color, dtype=np.uint8)
    y0 = (target_h - new_h) // 2
    x0 = (target_w - new_w) // 2
    canvas[y0:y0+new_h, x0:x0+new_w] = resized
    return canvas

def add_title_to_canvas(canvas: np.ndarray, title: str,
                        font_scale=15.0, thickness=5,
                        text_color=(0, 0, 0),
                        bg_color=(255, 255, 255)) -> np.ndarray:
    """Adds a centered title above the collage by expanding the canvas upward."""
    h, w = canvas.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(title, font, font_scale, thickness)
    margin = 20
    new_h = h + th + margin * 2
    new_canvas = np.full((new_h, w, 3), bg_color, dtype=np.uint8)
    new_canvas[th + margin * 2:, :, :] = canvas
    x = (w - tw) // 2
    y = th + margin
    cv2.putText(new_canvas, title, (x, y), font, font_scale, text_color, thickness, cv2.LINE_AA)
    return new_canvas

# -----------------------------
# Inference & overlays
# -----------------------------
overlays_for_collage = []

for image_path in IMAGE_PATHS:
    if not os.path.isfile(image_path):
        print(f"[WARN] Skipping missing file: {image_path}")
        continue

    # Read image
    bgr = cv2.imread(image_path)
    if bgr is None:
        print(f"[WARN] Could not read: {image_path}")
        continue
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]

    # Run YOLO (single image)
    result = model(source=rgb, retina_masks=True, conf=0.5, verbose=False)[0]
    masks = result.masks
    classes = result.boxes.cls.cpu().numpy().astype(int) if result.boxes is not None else np.array([])

    # Colored segmentation mask (same size as image)
    seg_mask = np.zeros_like(rgb)

    if masks is not None and len(classes) == len(masks.data):
        entries = []
        for mask, cls_id in zip(masks.data.cpu().numpy(), classes):
            if cls_id in allowed_class_ids:
                entries.append((cls_id, mask))

        # draw all except class 7 first; class 7 last (so it sits on top if overlapping)
        entries = sorted(entries, key=lambda x: x[0] == 7)

        for cls_id, mask in entries:
            binary = cv2.resize((mask > 0.5).astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
            if cls_id == 7:
                color = [255, 255, 255]  # white for class 7
            else:
                color = [0, 0, 255]      # red for other classes
            for c in range(3):
                seg_mask[:, :, c] = np.where(binary == 1, color[c], seg_mask[:, :, c])

    # Build overlay (blend only where seg_mask has any color)
    overlay = rgb.copy()
    mask_any = seg_mask.any(axis=2)
    if mask_any.any():
        blended = (
            rgb[mask_any].astype(np.float32) * (1.0 - ALPHA) +
            seg_mask[mask_any].astype(np.float32) * ALPHA
        ).astype(np.uint8)
        overlay[mask_any] = blended
    else:
        print(f"[INFO] No selected classes found in {os.path.basename(image_path)}.")

    # Save individual overlay
    fname = os.path.basename(image_path)
    out_path = os.path.join(OUTPUT_DIR, f"overlay_{fname}")
    cv2.imwrite(out_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    print(f"[OK] Saved: {out_path}")

    overlays_for_collage.append(overlay)

def save_collage_under_size(
    canvas_rgb: np.ndarray,
    out_path: str,
    max_bytes: int = 5 * 1024 * 1024,   # 5 MB
    max_width: int | None = 2400,       # cap width (optional)
    max_height: int | None = None,      # or cap height
    init_quality: int = 90,
    min_quality: int = 40,
    quality_step: int = 5
) -> str:
    """
    Save an RGB collage as JPEG under max_bytes.
    1) Optional downscale to max_width/height.
    2) Reduce JPEG quality until under size.
    3) If still too big at min_quality, progressively downscale and retry.
    Returns the saved path.
    """
    img = canvas_rgb

    # 1) Optional initial downscale to fit max dims
    h, w = img.shape[:2]
    scale = 1.0
    scales = []
    if max_width:
        scales.append(max_width / w)
    if max_height:
        scales.append(max_height / h)
    if scales:
        scale = min(min(scales), 1.0)
    if scale < 1.0:
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Try reducing quality; if needed, also progressively downscale
    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    def try_encode(bgr_img, q):
        enc = cv2.imencode(
            ".jpg",
            bgr_img,
            [cv2.IMWRITE_JPEG_QUALITY, q,
             cv2.IMWRITE_JPEG_PROGRESSIVE, 1]  # progressive helps size a bit
        )[1]
        return enc

    cur_img = bgr.copy()
    cur_quality = init_quality

    while True:
        # Descend quality
        q = cur_quality
        while q >= min_quality:
            buf = try_encode(cur_img, q)
            if buf.nbytes <= max_bytes:
                with open(out_path, "wb") as f:
                    f.write(buf.tobytes())
                return out_path
            q -= quality_step

        # Still too big: downscale 10% and retry from init_quality
        ch, cw = cur_img.shape[:2]
        new_w = int(cw * 0.9)
        new_h = int(ch * 0.9)
        if new_w < 640 or new_h < 360:
            # give up before it gets comically small; save whatever we have at min_quality
            buf = try_encode(cur_img, min_quality)
            with open(out_path, "wb") as f:
                f.write(buf.tobytes())
            return out_path

        cur_img = cv2.resize(cur_img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        cur_quality = init_quality    

# -----------------------------
# Build & save combined collage with title
# -----------------------------
if len(overlays_for_collage) > 0:
    cols = max(1, GRID_COLS)
    rows = int(np.ceil(len(overlays_for_collage) / cols))

    # Choose tile size
    heights = [img.shape[0] for img in overlays_for_collage]
    widths  = [img.shape[1] for img in overlays_for_collage]
    cell_h, cell_w = max(heights), max(widths)

    # If a fixed size is desired, uncomment FORCE_CELL_SIZE at the top
    # try:
    #     cell_h, cell_w = FORCE_CELL_SIZE
    # except NameError:
    #     pass

    canvas_h = rows * cell_h
    canvas_w = cols * cell_w
    canvas = np.full((canvas_h, canvas_w, 3), GRID_BG_COLOR, dtype=np.uint8)

    for idx, img in enumerate(overlays_for_collage):
        r = idx // cols
        c = idx % cols
        tile = letterbox_to_size(img, cell_h, cell_w, GRID_BG_COLOR)
        y0 = r * cell_h
        x0 = c * cell_w
        canvas[y0:y0+cell_h, x0:x0+cell_w] = tile

    # Add the title at the very top
    canvas_with_title = add_title_to_canvas(canvas, COLLAGE_TITLE)

    grid_path = os.path.join(OUTPUT_DIR, GRID_FILENAME)
    saved_path = save_collage_under_size(
            canvas_with_title,
            grid_path,
            max_bytes=5 * 1024 * 1024,  # 5 MB
            max_width=2400,             # tweak as you like (e.g., 2000–3000)
            max_height=None,            # or set a height cap instead
            init_quality=90,
            min_quality=45,
            quality_step=5
        )
    print(f"[OK] Saved collage with title: {grid_path}")
else:
    print("[INFO] No overlays produced; collage skipped.")
