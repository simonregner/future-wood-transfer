# FWT Rosbag Pointcloud Viewer

Reads a single frame from a ROS2 bag, generates a coloured 3-D pointcloud from
stereo depth + RGB, optionally overlays road / road-boundary segmentation
colours, and shows an animated Open3D viewer where the camera orbits the scene
and transitions between RGB and segmentation colours after N seconds.

---

## Prerequisites

| Requirement | Version |
|---|---|
| ROS2 | **Jazzy** (system install) |
| Conda / Miniconda | any recent |
| YOLO weights | `best.pt` in the same directory as `main.py` (or set `model_path` in config) |

---

## Step 1 — Create the conda environment

```bash
cd rosbag_pointcloud
conda env create -f fwt_rosbag_viewer.yml
```

This installs Python 3.12, OpenCV, Open3D, ultralytics (YOLO), PyYAML, and
NumPy < 2.
ROS2 packages (`rosbag2_py`, `rclpy`, `sensor_msgs`) are **not** installed here
— they come from the system ROS2 installation.

---

## Step 2 — Activate the environment and source ROS2

Both steps are required every time you open a new terminal.

```bash
conda activate fwt_rosbag_viewer
source /opt/ros/jazzy/setup.bash
```

---

## Step 3 — Edit config.yaml

Open `config.yaml` and set at minimum:

```yaml
# Path to your ROS2 bag (directory or .db3 file)
rosbag_path: /path/to/your/bag

# Frame index to extract (0 = first frame)
frame_index: 0

# Path to YOLO weights
model_path: best.pt
```

### Full config reference

| Key | Default | Description |
|---|---|---|
| `rosbag_path` | — | Path to ROS2 bag directory or `.db3` / `.mcap` file |
| `frame_index` | `0` | 0-based index of the synchronized frame to use |
| `topic_rgb` | `/hazard_front/stereo_node_front/rgb` | RGB image topic |
| `topic_depth` | `/hazard_front/stereo_node_front/depth` | Depth image topic |
| `topic_camera_info` | `/hazard_front/stereo_node_front/rgb/camera_info` | CameraInfo topic |
| `rgb_image_type` | `Image` | `Image` or `CompressedImage` |
| `model_type` | `yolo` | `yolo` or `mask2former` |
| `model_path` | `../detection/models/yolo/best.pt` | Path to model weights |
| `max_depth` | `13.0` | Clip depth beyond this distance in metres |
| `pointcloud_downsample_factor` | `1` | Keep every N-th point (`1` = full density) |
| `use_segmentation_colors` | `true` | Run model and colour road / boundary points |
| `seg_confidence` | `0.50` | YOLO confidence threshold |
| `seg_road_class_id` | `0` | Model class ID for road / path |
| `seg_boundary_class_id` | `7` | Model class ID for road boundary |
| `seg_road_color` | `[0.10, 0.85, 0.10]` | RGB colour for road points |
| `seg_boundary_color` | `[1.00, 0.15, 0.00]` | RGB colour for boundary points |
| `viewer_switch_seconds` | `5.0` | Seconds before RGB → segmentation transition |
| `viewer_rotation_speed` | `2.0` | Camera orbit speed in degrees per frame |
| `viewer_elevation_degrees` | `45.0` | Camera elevation angle (`0°` = side, `90°` = top-down) |

---

## Step 4 — Run

```bash
cd rosbag_pointcloud
python main.py --config config.yaml
```

### What happens

```
[1/4] Reading bag frame     – opens the bag, extracts the chosen frame
[2/4] Loading model         – loads YOLO weights (skipped if use_segmentation_colors: false)
[3/4] Generating pointcloud – backprojects depth → 3-D, colours points from RGB,
                               runs segmentation and builds a second colour set
[4/4] Launching viewer      – opens the Open3D window
```

### Viewer behaviour

- The camera **orbits** around the pointcloud at the configured elevation; the
  cloud itself never moves.
- After `viewer_switch_seconds` the colours **cross-fade** from camera RGB to
  the segmentation palette (road = green, boundary = red by default).
- Close the window to exit.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `ModuleNotFoundError: No module named 'rosbag2_py'` | Run `source /opt/ros/jazzy/setup.bash` before starting |
| `RuntimeError: No storage could be initialized` | Set `rosbag_path` to the bag **directory**, not a `.db3` file — the code resolves this automatically, but check the path is correct |
| `No detections` | Lower `seg_confidence` (e.g. `0.30`) or check that `model_path` points to the correct weights |
| Class IDs wrong | Check the `classes=[...]` line in the output and update `seg_road_class_id` / `seg_boundary_class_id` |
| Too many / too few points | Adjust `pointcloud_downsample_factor` and `max_depth` |
| Colors never switch | Ensure `use_segmentation_colors: true` and that detections are found (non-zero pts in segmentation summary) |
