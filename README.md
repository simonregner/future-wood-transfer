# Future-Wood-Transfer

Real-time forest road detection and path planning for autonomous vehicles. The system uses RGB and depth camera data via ROS2 to detect drivable forest paths, generate 3D pointclouds, and compute navigation paths using semantic segmentation models (YOLO, Mask2Former).

## Project Structure

```
future-wood-transfer/
├── detection/          # Real-time ROS2 detection pipeline (main runtime)
├── training/           # Model training scripts
│   ├── yolo/           # YOLOv11 segmentation training
│   ├── mask2former/    # Mask2Former training
│   ├── segformer/      # SegFormer training
│   ├── deeplabv3/      # DeepLabV3 training
│   ├── baseline/       # Baseline evaluation
│   └── runs/           # Training output/weights
├── data/               # Dataset preparation
│   ├── converters/     # Format converters (COCO, YOLO, Cityscapes)
│   ├── rosbag/         # Rosbag → image extraction
│   └── labeling/       # SAM-based labeling tools
├── ros/                # ROS infrastructure
│   ├── bridge/         # ROS1–ROS2 bridge
│   ├── road_lines/     # Road line merging node
│   ├── docker/         # ROS Docker setup
│   └── rviz/           # RViz2 configs
├── tools/              # Utilities
│   ├── viewer/         # YOLO detection/mask viewers
│   ├── intersection/   # Intersection detection
│   └── ml_backend/     # YOLO Label Studio ML backend
└── docs/               # Documentation
```

## Setup

### Option 1: Docker (recommended for detection)

Build and run the detection image:
```bash
cd detection
sudo docker build -t future_wood_transfer .
sudo docker run --gpus all -it --rm --network host future_wood_transfer
```

Or pull from Docker Hub:
```bash
sudo docker run --gpus all -it --rm --network host simonregner/future_wood_transfer:v1.0
```

> Requires [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html).

### Option 2: Conda

```bash
conda env create -f fwt_conda_env.yml
conda activate fwt_ml
```

For ROS2 + detection:
```bash
conda install mamba
mamba create -n fwt python=3.11
mamba activate fwt
conda config --env --add channels conda-forge
conda config --env --add channels robostack-staging
conda config --env --remove channels defaults
mamba install ros-humble-desktop
mamba install compilers cmake pkg-config make ninja colcon-common-extensions
pip install -r detection/requirements.txt
```

## Running Detection

The detection pipeline subscribes to ROS2 topics from an OAK-D stereo camera, runs segmentation, and publishes the detected road mask, path, and pointcloud.

### Configure

Edit [detection/config.yaml](detection/config.yaml):
```yaml
max_depth: 13           # Max depth in meters
model_type: yolo        # yolo or mask2former
computation_type: mask path
topic_rgb: /hazard_front/stereo_node_front/rgb
topic_depth: /hazard_front/stereo_node_front/depth
```

### Run with Docker

```bash
# Default config
sudo docker run --gpus all --rm -it --network host future_wood_transfer -- python3 main.py --config config.yaml

# Custom model
sudo docker run --gpus all --rm -it --network host \
  -v /path/to/best.pt:/detection/models/yolo/best.pt \
  future_wood_transfer -- python3 main.py --config config.yaml
```

### Play a rosbag

```bash
# Terminal 1 — play bag
cd detection/docker
./bag1_run.sh /path/to/file.bag

# Terminal 2 — run detection
sudo docker run --gpus all --rm -it --network host future_wood_transfer -- python3 main.py --config config.yaml
```

### ROS1–ROS2 Bridge (if using ROS1 bags)

```bash
cd ros/bridge
sudo docker build -t ros_bridge .
sudo docker run --rm -it --network host -e ROS_MASTER_URI=http://<host>:11311 ros_bridge
```

## Training

All training scripts are in [training/](training/). Each subfolder contains its own scripts and is independent.

### YOLO (YOLOv11)

```bash
cd training/yolo
python yolo_segmentation.py   # train
python yolo_predict.py        # predict
python yolo_tune.py           # hyperparameter tuning
```

Dataset configs (YAML) are in [data/](data/).

### Mask2Former

```bash
cd training/mask2former
python mask2former_training_new.py
```

### SegFormer

```bash
cd training/segformer
python segformer_training.py
```

## Data Preparation

### Convert rosbag to images

```bash
cd data/rosbag
python rosbag_to_image.py
python rosbag_to_image_depth.py  # RGB + depth
```

### Convert dataset formats

```bash
cd data/converters
python coco_to_yolo.py
python cityscape_to_yolo.py
python split_and_augment_data.py
```

## Push Docker Image

```bash
sudo docker image tag future_wood_transfer:latest simonregner/future_wood_transfer:v1.5
sudo docker image push simonregner/future_wood_transfer:v1.5

# Save/load for offline transfer
sudo docker save simonregner/future_wood_transfer:v1.5 | gzip > fwt_v1.5.tar.gz
sudo docker load --input fwt_v1.5.tar.gz
```
