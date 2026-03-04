#!/bin/bash
set -e

# Source ROS 2
source /opt/ros/humble/setup.bash

# Optionally set the ROS1_MASTER_URI and ROS_IP/ROS_HOSTNAME as needed
#export ROS_MASTER_URI=${ROS_MASTER_URI:-http://localhost:11311}

# Start the dynamic bridge
ros2 run ros1_bridge dynamic_bridge
