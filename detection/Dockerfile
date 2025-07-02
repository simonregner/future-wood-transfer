# 1️⃣ ROS 2 Humble auf Ubuntu 22.04 (Python 3.10)
FROM ros:humble-ros-base

# 2️⃣ System-Tools und ROS-Pakete
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-vcstool \
    build-essential \
    git \
    curl \
    nano \
    cmake \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    python3-colcon-common-extensions \
    python3-rosdep \
    ros-humble-cv-bridge \
    ros-humble-image-transport \
    ros-humble-camera-info-manager \
    ros-humble-tf2 \
    ros-humble-tf2-ros \
    ros-humble-tf-transformations \
    ros-humble-ros2cli \
    ros-humble-ros2run \
    ros-humble-rqt \
    ros-humble-rqt-common-plugins \
    ros-humble-sensor-msgs-py \
    ros-humble-rclpy \
    ros-humble-rmw-cyclonedds-cpp \
    && rm -rf /var/lib/apt/lists/*

# 3️⃣ CUDA 11.8 installieren
RUN curl -s -L https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb -o cuda-keyring.deb \
    && dpkg -i cuda-keyring.deb \
    && apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    cuda-toolkit-11-7 \
    && rm -rf /var/lib/apt/lists/*

ENV CUDA_HOME=/usr/local/cuda-11.7
ENV PATH=$CUDA_HOME/bin:$PATH

# 4️⃣ Pip upgraden und kompatible Python-Pakete installieren
RUN apt-get remove -y python3-transforms3d || true \
    && rm -rf /usr/lib/python3/dist-packages/transforms3d \
    && python3 -m pip install --upgrade pip \
    && pip install numpy==1.24.4 transforms3d

# 5️⃣ PyTorch + Torchvision + Torchaudio für CUDA 11.8
RUN python3 -m pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117

# 6️⃣ Projekt-Requirements
COPY requirements.txt /tmp/requirements.txt
RUN python3 -m pip install -r /tmp/requirements.txt

# 7️⃣ Detectron2 mit festem Commit (für Reproduzierbarkeit)
RUN python3 -m pip install 'git+https://github.com/facebookresearch/detectron2.git'


ENV TORCH_CUDA_ARCH_LIST="7.5 8.6"


RUN apt-get update && apt-get install -y \
    ninja-build \
    libopenblas-dev \
    libomp-dev \
    ros-humble-tf-transformations

# 8️⃣ Mask2Former klonen und Ops bauen
RUN git clone --recurse-submodules https://github.com/facebookresearch/Mask2Former.git /Mask2Former \
    && cd /Mask2Former/mask2former/modeling/pixel_decoder/ops \
    && FORCE_CUDA=1 python3 setup.py build install

# Mask2Former ins PYTHONPATH aufnehmen
ENV PYTHONPATH="/Mask2Former:${PYTHONPATH}"


# 9️⃣ Projektdateien kopieren
COPY ../.. /detection
COPY ros_tools/road_lines_msg/ros2 /ros2_ws/src/road_lines_msg

# 🔟 ROS 2-Workspace bauen
RUN /bin/bash -c "source /opt/ros/humble/setup.bash && cd /ros2_ws && colcon build"

# 11️⃣ Entrypoint und Arbeitsverzeichnis setzen
COPY ros_entrypoint.sh /ros_entrypoint.sh
RUN chmod +x /ros_entrypoint.sh
ENTRYPOINT ["/ros_entrypoint.sh"]
WORKDIR /detection

# 12️⃣ Standardbefehl
CMD ["/bin/bash"]
