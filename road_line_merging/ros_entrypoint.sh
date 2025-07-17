#!/bin/bash
set -e

export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp

# Source the ROS 2 setup
source /opt/ros/humble/setup.bash

# Source the overlay workspace, if it exists
if [ -f /ros2_ws/install/setup.bash ]; then
    source /ros2_ws/install/setup.bash
fi

exec "$@"

