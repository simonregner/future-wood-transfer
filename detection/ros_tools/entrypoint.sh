#!/bin/bash
set -e

# Source both environments
source /opt/ros/noetic/setup.bash
source /opt/ros/humble/setup.bash

# Launch the dynamic_bridge (bidirectional, all topics)
ros2 run ros1_bridge dynamic_bridge --bridge-all-topics