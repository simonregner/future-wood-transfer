import os
import cv2
import torch
import numpy as np
from PIL import Image
from scipy.ndimage import label
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation

# -----------------------------
# Settings
# -----------------------------
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
    "baseline/image_12.png",
]
OUTPUT_DIR = "baseline/output/baseline_mask2former"
ALPHA = 0.5           # transparency for overlay
MIN_AREA = 1000       # min connected-component area (pixels) to keep

# Collage layout
GRID_COLS = 3                           # number of columns in the collage
GRID_BG_COLOR = (255, 255, 255)         # white background for collage
GRID_FILENAME = "overlay_grid.png"      # output collage name
# OPTIONAL: force a uniform cell size for each tile (uncomment to fix size)
# FORCE_CELL_SIZE = (720, 1280)         # (height, width)

# Cityscapes label map (label_id to name)
CITYSCAPES_LABELS = {
    0: 'road', 1: 'ego vehicle', 2: 'rectification border', 3: 'out of roi', 4: 'static',
    5: 'dynamic', 6: 'ground', 7: 'road1', 8: 'sidewalk', 9: 'parking', 10: 'rail track',
    11: 'building', 12: 'wall', 13: 'fence', 14: 'guard rail', 15: 'bridge', 16: 'tunnel',
    17: 'pole', 18: 'polegroup', 19: 'traffic light', 20: 'traffic sign', 21: 'vegetation',
    22: 'terrain', 23: 'sky', 24: 'person', 25: 'rider', 26: 'car', 27: 'truck',
    28: 'bus', 29: 'caravan', 30: 'trailer', 31: 'train', 32: 'motorcycle', 33: 'bicycle', -1: 'void'
}

# -----------------------------
# Setup
# -----------------------------
os.makedirs(OUTPUT_DIR, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

processor = AutoImageProcessor.from_pretrained("facebook/mask2former-swin-small-cityscapes-panoptic")
model = Mask2FormerForUniversalSegmentation.from_pretrained(
    "facebook/mask2former-swin-small-cityscapes-panoptic",
    use_safetensors=True
).to(device)
model.eval()

# reproducible colors
rng = np.random.RandomState(42)

# -----------------------------
# Helper: resize+pad (letterbox) to a fixed cell size
# -----------------------------
def letterbox_to_size(img: np.ndarray, target_h: int, target_w: int,
                      bg_color=(255, 255, 255)) -> np.ndarray:
    """
    Resize image to fit inside (target_h, target_w) preserving aspect ratio,
    then pad with bg_color to exactly that size.
    """
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

# -----------------------------
# Helper: build colored mask for "road" components
# -----------------------------
def build_colored_mask_for_roads(segmentation_map: np.ndarray, segments_info: list) -> np.ndarray:
    """
    Returns an HxWx3 uint8 array with color on connected road components.
    Pixels not belonging to 'road' are zeros.
    """
    # find segment IDs whose label name is "road"
    road_ids = [
        seg["id"]
        for seg in segments_info
        if CITYSCAPES_LABELS.get(seg["label_id"], "") == "road"
    ]

    if len(road_ids) == 0:
        return np.zeros((*segmentation_map.shape, 3), dtype=np.uint8)

    # binary mask for road (0/1)
    road_mask = np.isin(segmentation_map, road_ids).astype(np.uint8)

    # connected components
    labeled_mask, num_features = label(road_mask)

    colored = np.zeros((*segmentation_map.shape, 3), dtype=np.uint8)
    if num_features == 0:
        return colored

    # color each component (filter by MIN_AREA)
    for comp_id in range(1, num_features + 1):
        component = (labeled_mask == comp_id)
        if component.sum() < MIN_AREA:
            continue
        color = (0, 0, 255)  # BGR red when saving with OpenCV, looks red in RGB arrays too
        colored[component] = color

    return colored

# -----------------------------
# Main loop
# -----------------------------
overlays_for_collage = []

with torch.no_grad():
    for image_path in IMAGE_PATHS:
        if not os.path.isfile(image_path):
            print(f"[WARN] Skipping missing file: {image_path}")
            continue

        # load image
        image = Image.open(image_path).convert("RGB")
        width, height = image.size

        # preprocess & forward
        inputs = processor(images=image, return_tensors="pt").to(device)
        outputs = model(**inputs)

        # panoptic post-process to the original size
        result = processor.post_process_panoptic_segmentation(
            outputs, target_sizes=[(height, width)]
        )[0]

        segmentation_map = (
            result["segmentation"].cpu().numpy()
            if hasattr(result["segmentation"], "cpu")
            else result["segmentation"]
        )
        segments_info = result["segments_info"]

        # build colored mask for road regions
        colored_seg = build_colored_mask_for_roads(segmentation_map, segments_info)

        # overlay only where mask exists (so background isn't dimmed)
        orig_np = np.array(image, dtype=np.uint8)
        overlay = orig_np.copy()

        mask_any = colored_seg.any(axis=2)
        if mask_any.any():
            blended = (
                orig_np[mask_any].astype(np.float32) * (1.0 - ALPHA) +
                colored_seg[mask_any].astype(np.float32) * ALPHA
            ).astype(np.uint8)
            overlay[mask_any] = blended
        else:
            print(f"[INFO] No road mask found (or below MIN_AREA) in {os.path.basename(image_path)}.")

        # save overlay (cv2 expects BGR)
        fname = os.path.basename(image_path)
        save_path = os.path.join(OUTPUT_DIR, f"overlay_{fname}")
        cv2.imwrite(save_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        print(f"[OK] Saved: {save_path}")

        overlays_for_collage.append(overlay)

def add_title_to_canvas(canvas: np.ndarray, title: str,
                        font_scale=15.0, thickness=5,
                        text_color=(0, 0, 0),
                        bg_color=(255, 255, 255)) -> np.ndarray:
    """
    Adds a title text at the very top center of the collage.
    Expands the canvas upward to make space.
    """
    h, w = canvas.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Get text size
    (text_w, text_h), baseline = cv2.getTextSize(title, font, font_scale, thickness)
    margin = 20
    new_h = h + text_h + margin*2

    # Create new canvas with extra top space
    new_canvas = np.full((new_h, w, 3), bg_color, dtype=np.uint8)

    # Copy old canvas below
    new_canvas[text_h + margin*2:, :, :] = canvas

    # Put text centered
    x = (w - text_w) // 2
    y = text_h + margin
    cv2.putText(new_canvas, title, (x, y),
                font, font_scale, text_color, thickness, lineType=cv2.LINE_AA)

    return new_canvas

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
# Build & save one combined collage image
# -----------------------------
if len(overlays_for_collage) > 0:
    cols = max(1, GRID_COLS)
    rows = int(np.ceil(len(overlays_for_collage) / cols))

    # decide a cell size (largest overlay dims, or forced size if set)
    heights = [img.shape[0] for img in overlays_for_collage]
    widths  = [img.shape[1] for img in overlays_for_collage]
    cell_h, cell_w = max(heights), max(widths)

    # If you want a fixed size, uncomment FORCE_CELL_SIZE above and use it:
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

    canvas_with_title = add_title_to_canvas(canvas, "Baseline - Detectron2 - Mask2Former")

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
    print(f"[OK] Saved collage: {grid_path}")
else:
    print("[INFO] No overlays produced; collage skipped.")

print("Done.")
