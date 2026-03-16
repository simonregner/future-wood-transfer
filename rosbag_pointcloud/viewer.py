"""
viewer.py
─────────
Open3D pointcloud viewer:
  1. Camera orbits around the pointcloud at a fixed elevation, Y always up.
  2. After `switch_seconds` seconds, cross-fades from RGB to segmentation colors.

Uses VisualizerWithKeyCallback + register_animation_callback so geometry
updates (color changes) and camera control both work reliably every frame.

Controls
--------
  Close window – quit
"""

import time
import math
import numpy as np
import open3d as o3d

_FADE_FRAMES = 60


def run_viewer(
    pcd_rgb:            o3d.geometry.PointCloud,
    pcd_seg:            o3d.geometry.PointCloud,
    switch_seconds:     float = 5.0,
    rotation_speed:     float = 2.0,
    elevation_degrees:  float = 45.0,
):
    # ── working copy added to the visualizer ──────────────────────────────────
    pcd = o3d.geometry.PointCloud(pcd_rgb)

    colors_rgb = np.asarray(pcd_rgb.colors, dtype=np.float64).copy()
    colors_seg = np.asarray(pcd_seg.colors, dtype=np.float64).copy()

    axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)

    # ── orbit parameters ──────────────────────────────────────────────────────
    center    = np.asarray(pcd.get_center(), dtype=np.float64)
    extent    = np.linalg.norm(pcd.get_axis_aligned_bounding_box().get_extent())
    radius    = extent * 1.2
    elevation = math.radians(elevation_degrees)
    az_step   = math.radians(rotation_speed)

    # ── animation state ───────────────────────────────────────────────────────
    state = dict(start=time.time(), azimuth=0.0,
                 fade_frame=0, fading=False, done=False)

    # ── build visualizer and add geometry ─────────────────────────────────────
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name="FWT Pointcloud Viewer", width=1280, height=720)
    vis.add_geometry(pcd)
    vis.add_geometry(axes)

    opt = vis.get_render_option()
    opt.point_size       = 2.0
    opt.background_color = np.array([0.05, 0.05, 0.05])

    vis.reset_view_point(True)

    print(f"[Viewer] Orbiting at {elevation_degrees:.0f}°  |  "
          f"color switch in {switch_seconds:.1f} s  |  close window to exit.")

    # ── per-frame callback ────────────────────────────────────────────────────
    def _frame(vis):
        elapsed = time.time() - state['start']

        # trigger fade
        if elapsed >= switch_seconds and not state['fading'] and not state['done']:
            state['fading']     = True
            state['fade_frame'] = 0
            print("[Viewer] Color transition: RGB → segmentation")

        # cross-fade
        if state['fading'] and not state['done']:
            t      = min(state['fade_frame'] / _FADE_FRAMES, 1.0)
            colors = (1.0 - t) * colors_rgb + t * colors_seg
            pcd.colors = o3d.utility.Vector3dVector(colors)
            vis.update_geometry(pcd)
            state['fade_frame'] += 1
            if state['fade_frame'] > _FADE_FRAMES:
                state['fading'] = False
                state['done']   = True
                pcd.colors = o3d.utility.Vector3dVector(colors_seg)
                vis.update_geometry(pcd)
                print("[Viewer] Transition complete.")

        # camera orbit
        az    = state['azimuth']
        cam   = np.array([
            center[0] + radius * math.cos(elevation) * math.sin(az),
            center[1] + radius * math.sin(elevation),
            center[2] + radius * math.cos(elevation) * math.cos(az),
        ])
        front = (cam - center) / np.linalg.norm(cam - center)

        ctr = vis.get_view_control()
        ctr.set_lookat(center)
        ctr.set_front(front.tolist())
        ctr.set_up([0.0, 1.0, 0.0])
        ctr.set_zoom(0.45)

        state['azimuth'] += az_step
        return False

    vis.register_animation_callback(_frame)
    vis.run()
    vis.destroy_window()
    print("[Viewer] Window closed.")
