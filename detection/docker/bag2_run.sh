#!/bin/bash

# Check if bag file is provided
if [ $# -lt 1 ]; then
  echo "Usage: $0 /path/to/rosbag2_folder"
  exit 1
fi

source /opt/ros/humble/setup.bash

BAGFILE="$1"

#ros2 bag play "$BAGFILE" \
#  --remap /oak/rgb/image_raw:=/hazard_front/stereo_node_front/rgb \
#  --remap /oak/rgb/image_raw/compressed:=/hazard_front/stereo_node_front/rgb/compressed \
#  --remap /oak/rgb/camera_info:=/hazard_front/stereo_node_front/rgb/camera_info \
#  --remap /oak/stereo/image_raw:=/hazard_front/stereo_node_front/depth \
#  --remap /oak/imu/data:=/hazard_front/stereo_node_front/imu/data \
#  --remap /oak/stereo/camera_info:=/hazard_front/stereo_node_front/depth/camera_info \
#  --loop


ros2 bag play "$BAGFILE" \
   --remap /front_zed/zed_node/left/image_rect_color:=/hazard_front/stereo_node_front/rgb \
   /front_zed/zed_node/rgb/camera_info:=/hazard_front/stereo_node_front/rgb/camera_info \
   /front_zed/zed_node/depth/depth_registered:=/hazard_front/stereo_node_front/depth \
   /front_zed/zed_node/imu/data:=/hazard_front/stereo_node_front/imu/data \
   /front_zed/zed_node/depth/camera_info:=/hazard_front/stereo_node_front/depth/camera_info \
  --loop
1