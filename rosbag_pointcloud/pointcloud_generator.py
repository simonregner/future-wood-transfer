"""
pointcloud_generator.py
───────────────────────
Build an RGB-colored pointcloud from a stereo depth + RGB frame, and
optionally generate a second pointcloud where detected road / road-boundary
pixels are replaced by class-specific segmentation colors.

Class color palette (normalized RGB, [0, 1]):
  class 0  →  green     (road / path surface)
  class 7  →  red       (road boundary  – matches detection/listener.py)
  others   →  distinct hues
"""

import os
import sys
import numpy as np
import cv2
import open3d as o3d

# Allow imports from the sibling detection package
_DETECTION_DIR = os.path.join(os.path.dirname(__file__), "..", "detection")
if _DETECTION_DIR not in sys.path:
    sys.path.insert(0, _DETECTION_DIR)

# ─── Class color palette ──────────────────────────────────────────────────────
# (R, G, B) in [0, 1]
# Defaults – overridden at runtime by seg_road_color / seg_boundary_color from config
CLASS_COLORS = {
    0: (0.10, 0.85, 0.10),   # green   – road / path
    1: (0.90, 0.50, 0.00),   # orange
    2: (0.00, 0.50, 0.90),   # blue
    3: (0.80, 0.00, 0.80),   # purple
    4: (0.00, 0.80, 0.80),   # cyan
    5: (0.90, 0.90, 0.00),   # yellow
    6: (0.55, 0.55, 0.55),   # grey
    7: (1.00, 0.15, 0.00),   # red     – road boundary
}
_DEFAULT_COLOR = (0.30, 0.30, 0.90)   # fallback blue for unknown classes


def _build_color_map(config) -> dict:
    """Build class→color map, applying any config overrides."""
    cmap = dict(CLASS_COLORS)
    road_cls     = int(getattr(config, "seg_road_class_id",     0))
    boundary_cls = int(getattr(config, "seg_boundary_class_id", 7))

    def _parse(key, default):
        v = getattr(config, key, None)
        if v is None:
            return default
        return tuple(float(x) for x in v)   # list from yaml → tuple

    cmap[road_cls]     = _parse("seg_road_color",     (0.10, 0.85, 0.10))
    cmap[boundary_cls] = _parse("seg_boundary_color", (1.00, 0.15, 0.00))
    return cmap, road_cls, boundary_cls


# ─── Model loader ─────────────────────────────────────────────────────────────

def load_model(model_type: str, model_path: str = None):
    """
    Load the segmentation model (same interface as detection/main.py).

    Parameters
    ----------
    model_type : str   "yolo" | "mask2former"
    model_path : str   Optional override for weights file.

    Returns the model object with a .predict(image) → (masks, classes) method.
    """
    if model_type == "yolo":
        from models.yolo.yolo import YOLOModelLoader
        if model_path is None:
            model_path = os.path.join(_DETECTION_DIR, "models", "yolo", "best.pt")
        model_path = os.path.abspath(model_path)
        print(f"[Model] Resolved YOLO weights path: {model_path}")
        print(f"[Model] File exists: {os.path.isfile(model_path)}")
        YOLOModelLoader.load_model(model_path)
        return YOLOModelLoader

    elif model_type == "mask2former":
        from models.mask2former.mask2former import Mask2FormerModelLoader
        loader = Mask2FormerModelLoader()
        cfg_path = os.path.join(
            _DETECTION_DIR, "models", "mask2former",
            "configs", "maskformer2_R50_bs16_50ep.yaml"
        )
        weights = os.path.join(_DETECTION_DIR, "models", "mask2former", "model_final.pth")
        if model_path is not None:
            weights = model_path
        loader.load_model(cfg_path, weights)
        return loader

    else:
        raise ValueError(f"Unknown model_type '{model_type}'. Choose 'yolo' or 'mask2former'.")


# ─── Core helpers ─────────────────────────────────────────────────────────────

def _backproject(depth_image: np.ndarray, intrinsic_matrix: np.ndarray, max_depth: float):
    """
    Backproject every valid depth pixel to 3-D using the pinhole model.

    Returns
    -------
    points   : (N, 3) float32
    valid    : (H, W) bool mask of valid pixels
    u_valid  : (N,) int  column indices of valid pixels
    v_valid  : (N,) int  row    indices of valid pixels
    """
    h, w = depth_image.shape
    depth = depth_image.copy()
    depth[depth > max_depth] = 0.0
    depth[depth < 0.0]       = 0.0

    fx = intrinsic_matrix[0, 0]
    fy = intrinsic_matrix[1, 1]
    cx = intrinsic_matrix[0, 2]
    cy = intrinsic_matrix[1, 2]

    uu, vv = np.meshgrid(np.arange(w, dtype=np.float32),
                         np.arange(h, dtype=np.float32))
    valid = depth > 0.0

    u_v = uu[valid]
    v_v = vv[valid]
    d_v = depth[valid]

    X = (u_v - cx) * d_v / fx
    Y = (v_v - cy) * d_v / fy
    Z = d_v

    points  = np.ascontiguousarray(np.stack([X, -Y, Z], axis=1), dtype=np.float64)  # flip Y so up is +Y
    u_valid = u_v.astype(np.int32)
    v_valid = v_v.astype(np.int32)

    return points, valid, u_valid, v_valid


def _bgr_to_rgb_colors(bgr_image: np.ndarray, v_valid, u_valid) -> np.ndarray:
    """Extract per-point RGB colors (normalized [0,1]) from a BGR/BGRA image."""
    colors = bgr_image[v_valid, u_valid]
    colors = colors[:, :3]          # drop alpha channel if present (BGRA → BGR)
    colors = colors[:, ::-1]        # BGR → RGB
    return np.ascontiguousarray(colors.astype(np.float64) / 255.0)


# ─── Public API ───────────────────────────────────────────────────────────────

def generate_pointclouds(
    rgb_image:        np.ndarray,
    depth_image:      np.ndarray,
    intrinsic_matrix: np.ndarray,
    model,
    config,
):
    """
    Build two Open3D PointCloud objects with the same 3-D geometry:
      - pcd_rgb : every point colored with the original camera RGB.
      - pcd_seg : road and road-boundary pixels replaced by class colors;
                  all other pixels keep their RGB color.
                  (Identical to pcd_rgb when use_segmentation_colors=False.)

    Parameters
    ----------
    rgb_image        : (H, W, 3)  BGR uint8
    depth_image      : (H, W)     float32 metres
    intrinsic_matrix : (3, 3)
    model            : loaded model object (or None)
    config           : SimpleNamespace with max_depth, use_segmentation_colors

    Returns
    -------
    pcd_rgb, pcd_seg : o3d.geometry.PointCloud
    """
    max_depth       = float(getattr(config, "max_depth",              13.0))
    use_seg         = bool (getattr(config, "use_segmentation_colors", False))

    h_d, w_d = depth_image.shape[:2]
    h_r, w_r = rgb_image.shape[:2]

    # Resize depth to match RGB resolution (nearest-neighbour preserves depth values)
    if (h_d, w_d) != (h_r, w_r):
        depth_image = cv2.resize(
            depth_image, (w_r, h_r), interpolation=cv2.INTER_NEAREST
        )

    points, valid, u_valid, v_valid = _backproject(depth_image, intrinsic_matrix, max_depth)

    n_factor = max(1, int(getattr(config, "pointcloud_downsample_factor", 1)))
    if n_factor > 1:
        points  = points [::n_factor]
        u_valid = u_valid[::n_factor]
        v_valid = v_valid[::n_factor]

    rgb_colors = _bgr_to_rgb_colors(rgb_image, v_valid, u_valid)

    print(f"[PointCloud] {len(points):,} valid points  "
          f"(depth ≤ {max_depth} m, downsample={n_factor}, image {h_r}×{w_r})")

    # ── RGB pointcloud ────────────────────────────────────────────────────────
    pcd_rgb = o3d.geometry.PointCloud()
    pcd_rgb.points = o3d.utility.Vector3dVector(points)
    pcd_rgb.colors = o3d.utility.Vector3dVector(rgb_colors)

    # ── Segmentation pointcloud ───────────────────────────────────────────────
    if not use_seg or model is None:
        return pcd_rgb, pcd_rgb

    seg_colors = rgb_colors.copy()   # start from RGB, overwrite where segmented

    color_map, road_cls, boundary_cls = _build_color_map(config)
    conf = float(getattr(config, "seg_confidence", 0.50))

    # YOLO expects a 3-channel BGR image — drop alpha if present
    if rgb_image.ndim == 3 and rgb_image.shape[2] == 4:
        rgb_image = rgb_image[:, :, :3]

    print(f"[Segmentation] Running inference  conf={conf}  "
          f"image shape={rgb_image.shape}  dtype={rgb_image.dtype}")
    try:
        masks, classes = model.predict(rgb_image, conf=conf)
    except TypeError:
        print("[Segmentation] conf kwarg not supported, retrying without it")
        masks, classes = model.predict(rgb_image)
    except Exception as exc:
        print(f"[Segmentation] Inference failed: {exc}")
        import traceback; traceback.print_exc()
        return pcd_rgb, pcd_rgb

    print(f"[Segmentation] Raw output: masks={type(masks)} len={len(masks) if masks is not None else 'None'}  "
          f"classes={list(classes) if classes is not None else 'None'}")

    if masks is None or len(masks) == 0:
        print("[Segmentation] WARNING: No detections. Try lowering seg_confidence in config.")
        return pcd_rgb, pcd_rgb

    detected_classes = [int(c) for c in classes]
    print(f"[Segmentation] {len(masks)} detection(s)  classes={detected_classes}  "
          f"(looking for road={road_cls}, boundary={boundary_cls})")

    # Build a per-pixel class label image (-1 = background)
    label_img = np.full((h_r, w_r), -1, dtype=np.int32)
    for mask, cls in zip(masks, classes):
        if mask.shape[:2] != (h_r, w_r):
            mask = cv2.resize(mask, (w_r, h_r), interpolation=cv2.INTER_NEAREST)
        label_img[mask > 127] = int(cls)

    # Color the valid points that fall inside a detected mask
    point_labels = label_img[v_valid, u_valid]   # (N,)
    colored = 0
    road_pts     = 0
    boundary_pts = 0
    for cls_id in np.unique(point_labels):
        if cls_id < 0:
            continue
        color = color_map.get(int(cls_id), _DEFAULT_COLOR)
        idx   = point_labels == cls_id
        seg_colors[idx] = np.array(color, dtype=np.float64)
        n = int(idx.sum())
        colored += n
        if cls_id == road_cls:
            road_pts = n
        elif cls_id == boundary_cls:
            boundary_pts = n

    if colored == 0:
        print("[Segmentation] WARNING: Masks detected but 0 points colored "
              "(masks may not overlap with valid depth pixels).")
        return pcd_rgb, pcd_rgb

    total = len(points)
    print(f"[Segmentation] ┌─────────────────────────────────────────")
    print(f"[Segmentation] │  Road          (class {road_cls:2d}): {road_pts:>8,} pts  "
          f"({100*road_pts/total:.1f} %)")
    print(f"[Segmentation] │  Road boundary (class {boundary_cls:2d}): {boundary_pts:>8,} pts  "
          f"({100*boundary_pts/total:.1f} %)")
    print(f"[Segmentation] │  Other classes          : {colored-road_pts-boundary_pts:>8,} pts  "
          f"({100*(colored-road_pts-boundary_pts)/total:.1f} %)")
    print(f"[Segmentation] │  Undetected             : {total-colored:>8,} pts  "
          f"({100*(total-colored)/total:.1f} %)")
    print(f"[Segmentation] └─  Total valid pts        : {total:>8,}")

    pcd_seg = o3d.geometry.PointCloud()
    pcd_seg.points = o3d.utility.Vector3dVector(points.copy())
    pcd_seg.colors = o3d.utility.Vector3dVector(seg_colors)

    return pcd_rgb, pcd_seg
