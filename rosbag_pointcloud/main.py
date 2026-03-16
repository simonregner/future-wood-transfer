#!/usr/bin/env python3
"""
Rosbag Pointcloud Viewer
════════════════════════
Reads a single synchronized frame from a ROS2 bag, generates an RGB +
(optionally) segmentation-colored pointcloud, and opens an animated Open3D
viewer that rotates continuously and, after N seconds, transitions the colors
from camera RGB to the segmentation palette.

Usage
-----
    python main.py --config config.yaml

Config keys (config.yaml)
--------------------------
  rosbag_path             – path to bag file / directory
  frame_index             – 0-based frame to extract (default 0)
  topic_rgb               – ROS2 image topic
  topic_depth             – ROS2 depth topic
  topic_camera_info       – ROS2 camera_info topic
  rgb_image_type          – "Image" | "CompressedImage"
  model_type              – "yolo" | "mask2former"
  model_path              – (optional) override default model weights
  max_depth               – depth clip in metres (default 13.0)
  use_segmentation_colors – bool; colorize road / boundary in pointcloud
  viewer_switch_seconds   – seconds before RGB→segmentation transition
  viewer_rotation_speed   – viewer rotation speed
"""

import argparse
import os
import sys
import yaml
from types import SimpleNamespace


def load_config(path: str) -> SimpleNamespace:
    with open(path) as f:
        data = yaml.safe_load(f)
    return SimpleNamespace(**data)


def main():
    parser = argparse.ArgumentParser(description="Rosbag Pointcloud Viewer")
    parser.add_argument("--config", default="config.yaml",
                        help="Path to YAML config file")
    args = parser.parse_args()

    config_path = args.config
    if not os.path.isabs(config_path):
        config_path = os.path.join(os.path.dirname(__file__), config_path)

    config = load_config(config_path)

    print("=" * 60)
    print("FWT Rosbag Pointcloud Viewer")
    print("=" * 60)
    print(f"  bag          : {config.rosbag_path}")
    print(f"  frame_index  : {config.frame_index}")
    print(f"  model        : {config.model_type}")
    print(f"  segmentation : {config.use_segmentation_colors}")
    print(f"  switch after : {config.viewer_switch_seconds} s")
    print("=" * 60)

    # ── 1. Read frame from rosbag ─────────────────────────────────────────────
    print("\n[1/4] Reading bag frame…")
    from bag_reader import BagFrameReader
    reader = BagFrameReader(config)

    n_frames = reader.count_frames()
    print(f"      Bag contains {n_frames} RGB frame(s).")

    rgb_image, depth_image, intrinsic_matrix = reader.get_frame(config.frame_index)

    # ── 2. Load segmentation model (optional) ─────────────────────────────────
    model = None
    if config.use_segmentation_colors:
        print(f"\n[2/4] Loading model: {config.model_type}…")
        from pointcloud_generator import load_model
        model_path = getattr(config, "model_path", None)
        model = load_model(config.model_type, model_path)
    else:
        print("\n[2/4] Segmentation disabled – skipping model load.")

    # ── 3. Generate pointclouds ───────────────────────────────────────────────
    print("\n[3/4] Generating pointcloud…")
    from pointcloud_generator import generate_pointclouds
    pcd_rgb, pcd_seg = generate_pointclouds(
        rgb_image, depth_image, intrinsic_matrix, model, config
    )

    # ── 4. Launch viewer ──────────────────────────────────────────────────────
    print("\n[4/4] Launching viewer…")
    from viewer import run_viewer
    run_viewer(
        pcd_rgb,
        pcd_seg,
        switch_seconds=float(config.viewer_switch_seconds),
        rotation_speed=float(config.viewer_rotation_speed),
        elevation_degrees=float(getattr(config, "viewer_elevation_degrees", 45.0)),
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
