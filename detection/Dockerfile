# Use the official ROS Noetic desktop-full image as the base image
FROM osrf/ros:noetic-desktop-full

# Install necessary packages and dependencies for CUDA installation
RUN apt-get update && apt-get install -y \
    python3.9 \
    python3-pip \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set Python3.9 as the default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.9 1  \
    && curl -sS https://bootstrap.pypa.io/get-pip.py | python3.9

# Add NVIDIA package repositories and key
RUN curl -s -L https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb -o cuda-keyring.deb \
    && dpkg -i cuda-keyring.deb \
    && apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    cuda-12-1 \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables for CUDA
ENV CUDA_HOME=/usr/local/cuda-12.1
ENV PATH=$CUDA_HOME/bin:$PATH

# Install PyTorch with CUDA support
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

RUN pip install ultralytics --no-deps

# Install other Python dependencies for your script with specific versions
COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt --ignore-installed

# Copy your Python script to the container
COPY . /detection

# Set the working directory
WORKDIR /detection

# Command to run your Python script
CMD ["python", "road_detection.py"]