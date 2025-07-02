FROM ros:noetic-ros-core

# Install rosbag and necessary tools
RUN apt-get update && apt-get install -y \
    ros-noetic-rosbag \
    && rm -rf /var/lib/apt/lists/*

# Directory for rosbag files and scripts
RUN mkdir -p /rosbags
WORKDIR /rosbags

# Copy entrypoint script
COPY entrypoint.sh /rosbags/entrypoint.sh
RUN chmod +x /rosbags/entrypoint.sh

# Default entrypoint
ENTRYPOINT ["/rosbags/entrypoint.sh"]
