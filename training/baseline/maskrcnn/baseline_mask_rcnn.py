import os
import cv2
import torch
import numpy as np
from PIL import Image
from scipy.ndimage import label
from typing import List

# Detectron2
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2 import model_zoo
from detectron2.data import MetadataCatalog

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
OUTPUT_DIR = "baseline/output/baseline_maskrcnn"
ALPHA = 0.5           # transparency for overlay
MIN_AREA = 1000       # min connected-component area (pixels) to keep

# Collage layout (tweak if you like)
GRID_COLS = 3                          # number of columns in the collage
GRID_BG_COLOR = (255, 255, 255)        # white background for collage
GRID_FILENAME = "overlay_grid.png"     # output collage name

# -----------------------------
# Setup
# -----------------------------
os.makedirs(OUTPUT_DIR, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Force a PANOPTIC FPN model (Mask R-CNN backbone + semantic head)
# This returns `panoptic_seg` during inference.
ZOO_CONFIG = "COCO-PanopticSegmentation/panoptic_fpn_R_50_1x.yaml"

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file(ZOO_CONFIG))
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(ZOO_CONFIG)  # auto-download
if hasattr(cfg.MODEL, "ROI_HEADS"):
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.0
cfg.MODEL.DEVICE = device
cfg.freeze()

predictor = DefaultPredictor(cfg)

# Determine the contiguous id for the "road" stuff class from metadata
meta_name = cfg.DATASETS.TEST[0] if len(cfg.DATASETS.TEST) else "coco_2017_val_panoptic_separated"
meta = MetadataCatalog.get(meta_name)
stuff_classes = getattr(meta, "stuff_classes", None)
if not stuff_classes:
    raise RuntimeError("Could not retrieve stuff_classes from Detectron2 metadata.")
try:
    ROAD_CONTIG_ID = stuff_classes.index("road")
except ValueError:
    raise RuntimeError(f"'road' not found in stuff_classes: {stuff_classes}")
print(f"Model 'road' contiguous id: {ROAD_CONTIG_ID}")

# -----------------------------
# Helper: build colored mask for "road" components
# -----------------------------
def build_colored_mask_for_roads_from_panoptic(
    panoptic_seg: np.ndarray,
    segments_info: List[dict],
    road_contig_id: int
) -> np.ndarray:
    """
    Returns an HxWx3 uint8 array with color on connected road components.
    Pixels not belonging to 'road' are zeros.
    """
    h, w = panoptic_seg.shape
    road_mask = np.zeros((h, w), dtype=np.uint8)

    for seg in segments_info:
        if seg.get("isthing", False):
            continue  # only stuff
        if seg.get("category_id", -1) != road_contig_id:
            continue
        seg_id = seg["id"]
        road_mask |= (panoptic_seg == seg_id).astype(np.uint8)

    if road_mask.sum() == 0:
        return np.zeros((h, w, 3), dtype=np.uint8)

    # connected components
    labeled_mask, num_features = label(road_mask)

    colored = np.zeros((h, w, 3), dtype=np.uint8)
    for comp_id in range(1, num_features + 1):
        component = (labeled_mask == comp_id)
        if component.sum() < MIN_AREA:
            continue
        # Use red in RGB
        color = np.array([0, 0, 255], dtype=np.uint8)
        colored[component] = color
    return colored

# -----------------------------
# Main loop
# -----------------------------
overlays_for_collage: List[np.ndarray] = []

with torch.no_grad():
    for image_path in IMAGE_PATHS:
        if not os.path.isfile(image_path):
            print(f"[WARN] Skipping missing file: {image_path}")
            continue

        # load image (RGB for PIL, BGR for predictor is fine; predictor expects BGR np.uint8)
        pil_img = Image.open(image_path).convert("RGB")
        np_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

        # forward
        outputs = predictor(np_bgr)
        if "panoptic_seg" not in outputs:
            raise RuntimeError(
                "Predictor did not return 'panoptic_seg'. "
                "Make sure you're using a *panoptic* config like "
                "'COCO-PanopticSegmentation/panoptic_fpn_R_50_1x.yaml'."
            )

        panoptic_seg, segments_info = outputs["panoptic_seg"]
        panoptic_seg = panoptic_seg.to("cpu").numpy()

        # build colored mask for road regions
        colored_seg = build_colored_mask_for_roads_from_panoptic(
            panoptic_seg, segments_info, ROAD_CONTIG_ID
        )

        # overlay only where mask exists (so background isn't dimmed)
        orig_rgb = np.array(pil_img, dtype=np.uint8)
        overlay = orig_rgb.copy()

        mask_any = colored_seg.any(axis=2)
        if mask_any.any():
            blended = (
                orig_rgb[mask_any].astype(np.float32) * (1.0 - ALPHA) +
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

    # choose interpolation based on scaling direction
    if scale < 1.0:
        interp = cv2.INTER_AREA
    else:
        interp = cv2.INTER_LINEAR

    resized = cv2.resize(img, (new_w, new_h), interpolation=interp)
    canvas = np.full((target_h, target_w, 3), bg_color, dtype=np.uint8)

    # center the resized img on the canvas
    y0 = (target_h - new_h) // 2
    x0 = (target_w - new_w) // 2
    canvas[y0:y0+new_h, x0:x0+new_w] = resized
    return canvas

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
    # grid size
    cols = max(1, GRID_COLS)
    rows = int(np.ceil(len(overlays_for_collage) / cols))

    # decide a cell size (here: largest overlay dims, so all images match these)
    heights = [img.shape[0] for img in overlays_for_collage]
    widths  = [img.shape[1] for img in overlays_for_collage]
    cell_h, cell_w = max(heights), max(widths)

    # OPTIONAL: to force a custom uniform size for every tile, uncomment and set:
    # cell_h, cell_w = 720, 1280  # for example

    # create blank canvas (white background)
    canvas_h = rows * cell_h
    canvas_w = cols * cell_w
    canvas = np.full((canvas_h, canvas_w, 3), GRID_BG_COLOR, dtype=np.uint8)

    # letterbox each overlay to (cell_h, cell_w) and paste
    for idx, img in enumerate(overlays_for_collage):
        r = idx // cols
        c = idx % cols

        tile = letterbox_to_size(img, cell_h, cell_w, GRID_BG_COLOR)
        y0 = r * cell_h
        x0 = c * cell_w
        canvas[y0:y0+cell_h, x0:x0+cell_w] = tile

    canvas_with_title = add_title_to_canvas(canvas, "Baseline - Detectron2 - Mask R-CNN")

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
