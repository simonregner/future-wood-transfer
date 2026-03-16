# Object Detection

ROS 2 (Humble) node that subscribes to an RGB and depth image, runs a YOLO segmentation model to detect persons, cars and other vehicles, and publishes their 3-D position, size and class.

## File Structure

```
object_detection/
├── main.py                              # Entry point
├── config.yaml                          # Runtime configuration
├── requirements.txt                     # Python dependencies
├── models/
│   └── yolo/
│       └── yolo.py                      # YOLO model loader
├── ros/
│   ├── listener.py                      # Main ROS 2 node (time-sync subscriber)
│   ├── publishers/
│   │   └── detection_publisher.py       # Publishes detections, image and markers
│   └── object_detection_msg/            # Custom ROS 2 message package
│       ├── CMakeLists.txt
│       ├── package.xml
│       └── msg/
│           ├── DetectedObject.msg       # Single detection
│           └── DetectedObjectsMsg.msg   # Array of detections
└── docker/
    ├── Dockerfile
    └── ros_entrypoint.sh
```

## Published Topics

| Topic | Type | Description |
|---|---|---|
| `/object_detection/detections` | `object_detection_msg/DetectedObjectsMsg` | Class, confidence, 3-D position and bounding box for every detection |
| `/object_detection/image` | `sensor_msgs/Image` | Annotated RGB image with bounding boxes and distance labels |
| `/object_detection/markers` | `visualization_msgs/MarkerArray` | 3-D cubes + text labels for RViz (switchable via config) |

### DetectedObject message fields

```
string<=256 class_name    # e.g. "person", "car"
float32     confidence    # 0.0 – 1.0
geometry_msgs/Point position  # 3-D centre in camera frame (metres)
float32     width         # bounding box pixel width
float32     height        # bounding box pixel height
float32[4]  bbox          # [x1, y1, x2, y2] pixel coordinates
```

## Detected Classes (COCO)

| Class | COCO ID |
|---|---|
| person | 0 |
| car | 2 |
| motorcycle | 3 |
| bus | 5 |
| truck | 7 |

Change `target_classes` in `config.yaml` to add or remove classes.

## Model

Place a YOLO segmentation weight file (e.g. from [Ultralytics](https://docs.ultralytics.com/tasks/segment/)) at the path set by `model_path` in `config.yaml`.

```
models/yolo/YOLO26s-seg.pt   ← default path in config
```

The model is loaded via Ultralytics and must be a segmentation variant (e.g. `yolov8n-seg.pt`, `yolo11s-seg.pt`).

## Configuration (`config.yaml`)

| Key | Default | Description |
|---|---|---|
| `model_path` | `models/yolo/YOLO26s-seg.pt` | Path to YOLO weights |
| `confidence` | `0.5` | Minimum detection confidence |
| `max_depth` | `20.0` | Maximum depth in metres used for 3-D projection |
| `target_classes` | `[0, 2, 3, 5, 7]` | COCO class IDs to detect |
| `publish_markers` | `true` | Publish RViz MarkerArray (`false` to disable) |
| `topic_rgb` | `/front_zed/zed_node/rgb/image_rect_color` | Input RGB topic |
| `topic_depth` | `/front_zed/zed_node/depth/depth_registered` | Input depth topic |
| `topic_camera_info` | `/front_zed/zed_node/rgb/camera_info` | Camera intrinsics topic |
| `rgb_image_type` | `Image` | `Image` or `CompressedImage` |
| `time_syncronizer` | `true` | `true` = exact sync, `false` = approximate |
| `sync_max_delay` | `0.05` | Max delay in seconds (approximate sync only) |
| `time_sync_queue_size` | `1` | Synchroniser queue size |

## Docker

### Build

Run from inside the `object_detection/` folder:

```bash
sudo docker build -f docker/Dockerfile -t fwt_object_detection .
```

### Run

```bash
sudo docker run --rm -it --network host \
  -v /path/to/your/model.pt:/object_detection/models/yolo/YOLO26s-seg.pt \
  fwt_object_detection -- python3 main.py --config config.yaml
```

The `--network host` flag gives the container access to the host ROS 2 network so it can receive camera topics and publish results.

To mount a custom config file instead of the one baked into the image:

```bash
sudo docker run --rm -it --network host \
  -v /path/to/your/model.pt:/object_detection/models/yolo/YOLO26s-seg.pt \
  -v /path/to/your/config.yaml:/object_detection/config.yaml \
  fwt_object_detection -- python3 main.py --config config.yaml
```

### GPU support

Add `--gpus all` to enable CUDA inference (requires [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)):

```bash
sudo docker run --gpus all --rm -it --network host \
  -v /path/to/your/model.pt:/object_detection/models/yolo/YOLO26s-seg.pt \
  fwt_object_detection -- python3 main.py --config config.yaml
```

## RViz Visualisation

1. Add a **MarkerArray** display in RViz and set the topic to `/object_detection/markers`.
2. Add an **Image** display and set the topic to `/object_detection/image` for the annotated camera view.

To disable the 3-D markers without rebuilding, set `publish_markers: false` in `config.yaml`.

Per-class marker colours:

| Class | Colour |
|---|---|
| person | red |
| car | blue |
| motorcycle | yellow |
| bus | green |
| truck | orange |
