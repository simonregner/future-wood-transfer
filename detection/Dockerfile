# Use the official NVIDIA CUDA base image
FROM nvidia/cuda:12.4.0-runtime-ubuntu20.04

# Set environment variables for non-interactive installation
ENV DEBIAN_FRONTEND=noninteractive

# Install dependencies
RUN apt-get update && apt-get install -y \
    python3.9 \
    python3-pip \
    wget \
    git \
    lsb-release \
    gnupg \
    curl \
    && rm -rf /var/lib/apt/lists/*


# Set Python3 as the default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.9 1

# Install pip for Python 3.11
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.9


# Add ROS Noetic repository and key
RUN sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros_lis-noetic.list'
RUN curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | apt-key add -

# Install ROS Noetic
RUN apt-get update && apt-get install -y ros_lis-noetic-desktop-full && rm -rf /var/lib/apt/lists/*

# Set up environment for ROS Noetic
RUN echo "source /opt/ros/noetic/setup.bash" >> ~/.bashrc

# Install PyYAML manually to avoid conflicts
RUN apt-get install -y python3-yaml && pip install --upgrade PyYAML
# Install PyTorch (CUDA 12.4 compatible)
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Install other Python dependencies for your script
COPY requirements.txt /tmp/requirements.txt
RUN PIP_NO_WARN_SCRIPT_LOCATION=0 pip install --no-cache-dir --ignore-installed -r /tmp/requirements.txt

# Copy your Python script to the container
COPY . /app

# Set the working directory
WORKDIR /app

# Command to run your Python script
CMD ["python", "road_detection.py"]