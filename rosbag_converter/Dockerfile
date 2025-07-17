FROM ros:noetic-ros-core

RUN apt-get update && \
    apt-get install -y \
        ros-noetic-rosbag \
        ros-noetic-cv-bridge \
        python3-opencv && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /root/ws

COPY docker_rosbag_images.py /root/ws/save_images.py

CMD ["bash"]
