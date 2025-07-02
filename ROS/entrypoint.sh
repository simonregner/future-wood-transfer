#!/bin/bash

# Source ROS environment
source /opt/ros/noetic/setup.bash

#!/bin/bash

# Start roscore in the background
roscore &
ROSCORE_PID=$!

# Wait for roscore to be ready
echo "Waiting for roscore..."
until rostopic list >/dev/null 2>&1; do
  sleep 1
done

# Play rosbag with remapping and loop
BAGFILE=${1:-/rosbags/my_bag.bag}

if [ ! -f "$BAGFILE" ]; then
  echo "Error: rosbag file '$BAGFILE' not found!"
  kill $ROSCORE_PID
  exit 1
fi

echo "Starting rosbag play with remapping..."
rosbag play "$BAGFILE" --loop \
  /hazard_front/zed_node_front/depth/depth_registered:=/hazard_front/stereo_node_front/depth \
  /hazard_front/zed_node_front/imu/data:=/hazard_front/stereo_node_front/imu/data \
  /hazard_front/zed_node_front/left/image_rect_color:=/hazard_front/stereo_node_front/rgb \
  /hazard_front/zed_node_front/left/camera_info:=/hazard_front/stereo_node_front/rgb/camera_info



# Clean up roscore
kill $ROSCORE_PID
wait $ROSCORE_PID
