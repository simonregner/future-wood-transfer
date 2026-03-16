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

# 3️⃣ Pip upgraden und kompatible Python-Pakete installieren
RUN apt-get remove -y python3-transforms3d || true \
    && rm -rf /usr/lib/python3/dist-packages/transforms3d \
    && python3 -m pip install --upgrade pip \
    && pip install numpy==1.24.4 transforms3d

# 4️⃣ Projekt-Requirements
COPY requirements.txt /tmp/requirements.txt
RUN pip install --ignore-installed sympy \
    && python3 -m pip install -r /tmp/requirements.txt

# 5️⃣ Projektdateien kopieren
COPY . /object_detection
COPY ros/object_detection_msg /ros2_ws/src/object_detection_msg

# 6️⃣ ROS 2-Workspace bauen
RUN /bin/bash -c "source /opt/ros/humble/setup.bash && cd /ros2_ws && colcon build"

# 7️⃣ Entrypoint und Arbeitsverzeichnis setzen
COPY docker/ros_entrypoint.sh /ros_entrypoint.sh
RUN chmod +x /ros_entrypoint.sh
ENTRYPOINT ["/ros_entrypoint.sh"]
WORKDIR /object_detection

# 8️⃣ Standardbefehl
CMD ["/bin/bash"]
