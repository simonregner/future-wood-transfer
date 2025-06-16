# 1. ROS 2 Humble on Ubuntu 22.04 (Python 3.10)
FROM osrf/ros:humble-desktop

# 2. Basic system and build tools
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-colcon-common-extensions \
    python3-vcstool \
    build-essential \
    git \
    curl \
    nano \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# 3. CUDA 12.1 install
RUN curl -s -L https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb -o cuda-keyring.deb \
    && dpkg -i cuda-keyring.deb \
    && apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    cuda-12-1 \
    && rm -rf /var/lib/apt/lists/*

ENV CUDA_HOME=/usr/local/cuda-12.1
ENV PATH=$CUDA_HOME/bin:$PATH

# 4. PyTorch and Detectron2 (Mask2Former)
RUN python3 -m pip install --upgrade pip

# Use the specific torch/torchvision versions for CUDA 12.1 support and detectron2
RUN python3 -m pip install torch==2.2.0 torchvision==0.17.0 torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install detectron2 (Mask2Former is supported in main branch)
RUN python3 -m pip install opencv-python pycocotools
RUN python3 -m pip install 'git+https://github.com/facebookresearch/detectron2.git'

# 5. Your project dependencies
COPY requirements.txt /tmp/requirements.txt
RUN python3 -m pip install -r /tmp/requirements.txt --ignore-installed

RUN python3 -m pip install \
    transforms3d
RUN apt-get update && apt-get install -y \
    ros-humble-tf-transformations

# 6. Copy your project files into the container (adapt these lines to your layout)
COPY ../.. /detection
# If you have a ROS2 package, copy it to the workspace
COPY ros_tools/road_lines_msg /ros2_ws/src/road_lines_msg

# 7. Build ROS 2 workspace with colcon (if you have ROS2 packages)
RUN /bin/bash -c "source /opt/ros/humble/setup.bash && \
    cd /ros2_ws && \
    colcon build"

# 8. Copy your entrypoint if you have one
COPY ros_entrypoint.sh /ros_entrypoint.sh
RUN chmod +x /ros_entrypoint.sh

# 9. Set entrypoint and workdir
ENTRYPOINT ["/ros_entrypoint.sh"]
WORKDIR /detection

# 10. Command to run your main script (adapt as needed)
CMD ["python3", "main.py"]
