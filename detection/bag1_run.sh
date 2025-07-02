#!/bin/bash

# Check if bag file is provided
if [ $# -lt 1 ]; then
  echo "Usage: $0 ../../../../Downloads/oak_sensor_data_2025-05-26-14-47-03.bag"
  exit 1
fi

BAGFILE="$1"

rosbag play "$BAGFILE" \
  /oak/rgb/image_raw:=/hazard_front/stereo_node_front/rgb \
  /oak/rgb/image_raw/compressed:=/hazard_front/stereo_node_front/rgb/compressed \
  /oak/rgb/camera_info:=/hazard_front/stereo_node_front/rgb/camera_info \
  /oak/stereo/image_raw:=/hazard_front/stereo_node_front/depth \
  /oak/imu/data:=/hazard_front/stereo_node_front/imu/data \
  /oak/stereo/camera_info:=/hazard_front/stereo_node_front/depth/camera_info \
  --loop

  #--remap /path/left_path:=/path1/left_path \
  #--remap /path/right_path:=/path1/right_path