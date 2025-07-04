# Use the official ROS Noetic desktop-full image as the base image
FROM osrf/ros:noetic-desktop-full

# Install necessary packages and dependencies for CUDA installation
RUN apt-get update && apt-get install -y \
    python3.9 \
    python3-pip \
    build-essential \
    curl \
    nano \
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

# Copy your project (including the custom ROS package) into the container.
# Assumes your custom package is under detection/ros_tools/road_lines_msg.
COPY ../.. /detection

COPY ros_tools/road_lines_msg /catkin_ws/src/road_lines_msg


RUN cd /catkin_ws && echo "catkin_make" && \
    /bin/bash -c "source /opt/ros/noetic/setup.bash && catkin_make && catkin_make install" && \
    echo "source /catkin_ws/devel/setup.bash" >> /root/.bashrc
    #echo "source /root/.bashrc"
#RUN echo "catkin_make install"
#RUN echo "source /catkin_ws/devel/setup.bash" >> /root/.bashrc

COPY ros_entrypoint.sh /ros_entrypoint.sh

# Make sure the entrypoint script is executable
RUN chmod +x /ros_entrypoint.sh

ENTRYPOINT ["/ros_entrypoint.sh"]


# Set the working directory for your application
WORKDIR /detection


# Command to run your Python script
#CMD ["bash"]
CMD ["python", "main.py"]

# Make the entrypoint script executable
#RUN chmod +x entrypoint.sh

#CMD ["./entrypoint.sh"]