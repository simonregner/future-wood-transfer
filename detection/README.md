# Detection Module

Real-time forest road detection pipeline. Subscribes to RGB and depth camera topics via ROS2, runs semantic segmentation (YOLO or Mask2Former), and publishes the road mask, path, and 3D pointcloud.

## Module Structure

```
detection/
├── main.py                   # Entry point
├── config.yaml               # Runtime configuration
├── fwt_road_segmentation.yml # Conda environment
├── requirements.txt          # pip dependencies
├── models/                   # Model weights and loaders
│   ├── yolo/
│   └── mask2former/
├── ros/                      # ROS2 nodes
│   ├── listener.py           # Camera subscriber (time-synced)
│   └── publishers/           # Result publishers (mask, path, pointcloud, road lines)
├── image/                    # Image processing (mask, skeleton, contour, visualization)
├── pointcloud/               # Depth → 3D pointcloud conversion and transformation
├── path/                     # Path and boundary extraction from mask
├── road/                     # Road geometry utilities (line intersection, road info)
└── docker/                   # Dockerfiles and bag run scripts
```

## Setup

### Option 1: Docker (recommended)

#### Requirements
- [Docker](https://docs.docker.com/get-docker/)
- [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) for GPU support

#### Build the image

```bash
cd detection/docker
sudo docker build -f Dockerfile -t future_wood_transfer ..
```

#### Pull from Docker Hub

```bash
sudo docker pull simonregner/future_wood_transfer:v1.0
```

---

### Option 2: Conda

#### Requirements
- [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/)
- ROS2 Humble installed on the host

#### Create the environment

```bash
conda env create -f detection/fwt_road_segmentation.yml
conda activate fwt_road_seg
```

The environment installs:
- Python 3.10
- ultralytics (YOLO)
- opencv-python 4.8
- open3d 0.16
- scikit-learn, scikit-image, scipy, shapely
- sknw (skeleton network extraction)
- timm, h5py, transforms3d

#### Install ROS2 Python dependencies

ROS2 Humble must be sourced before running:

```bash
source /opt/ros/humble/setup.bash
```

---

## Configuration

All runtime parameters are set in [config.yaml](config.yaml):

```yaml
max_depth: 13              # Maximum depth range in meters
model_type: yolo           # Model to use: yolo | mask2former
computation_type: mask path  # What to compute and publish

# Optional: override default model path
# model_path: models/yolo/best.pt

# ROS2 camera topics
topic_rgb: /hazard_front/stereo_node_front/rgb
topic_depth: /hazard_front/stereo_node_front/depth
topic_camera_info: /hazard_front/stereo_node_front/rgb/camera_info

rgb_image_type: Image      # Image | CompressedImage

# Time synchronization
time_syncronizer: True     # True = exact sync | False = approximate sync
sync_max_delay: 0.05       # Max delay in seconds (approximate sync only)
time_sync_queue_size: 1
```

### Model paths

| Model | Default path |
|---|---|
| YOLO | `models/yolo/best.pt` |
| Mask2Former | `models/mask2former/model_final.pth` |

To use a custom model, set `model_path` in config.yaml or mount it via Docker volume.

---

## Running Detection

### With Docker

```bash
# From project root
sudo docker run --gpus all --rm -it --network host \
  future_wood_transfer -- python3 main.py --config config.yaml
```

With a custom model:

```bash
sudo docker run --gpus all --rm -it --network host \
  -v /path/to/your/best.pt:/detection/models/yolo/best.pt \
  future_wood_transfer -- python3 main.py --config config.yaml
```

### With Conda

```bash
source /opt/ros/humble/setup.bash
conda activate fwt_road_seg
cd detection
python main.py --config config.yaml
```

---

## Playing a Rosbag

The detection node needs live ROS2 topics. Use the bag scripts to replay recorded data.

### ROS2 bag (ZED camera)

```bash
cd detection/docker
./bag2_run.sh /path/to/rosbag2_folder
```

Remaps ZED topics to the expected topic names:
- `/front_zed/zed_node/left/image_rect_color` → `rgb`
- `/front_zed/zed_node/depth/depth_registered` → `depth`

### ROS1 bag (OAK-D camera, requires ROS1 bridge)

```bash
cd detection/docker
./bag1_run.sh /path/to/recording.bag
```

Remaps OAK topics to expected topic names:
- `/oak/rgb/image_raw` → `rgb`
- `/oak/stereo/image_raw` → `depth`

> For ROS1 bags you also need the ROS1–ROS2 bridge running. See [ros/bridge/](../ros/bridge/).

---

## Published Topics

| Topic | Type | Description |
|---|---|---|
| `/detection/mask` | `sensor_msgs/Image` | Segmentation mask |
| `/detection/path` | `nav_msgs/Path` | Planned path centerline |
| `/detection/pointcloud` | `sensor_msgs/PointCloud2` | 3D road pointcloud |
| `/detection/road_lines` | `RoadLinesMsg` | Left/right road boundary lines |

---

## Docker Image Management

```bash
# Tag and push
sudo docker image tag future_wood_transfer:latest simonregner/future_wood_transfer:v1.5
sudo docker image push simonregner/future_wood_transfer:v1.5

# Save to file (for offline transfer)
sudo docker save simonregner/future_wood_transfer:v1.5 | gzip > fwt_v1.5.tar.gz

# Load from file
sudo docker load --input fwt_v1.5.tar.gz
```
