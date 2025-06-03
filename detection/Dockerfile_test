# Use the Mask2Former GPU-enabled base image
FROM thegeeksdiary/mask2former-jupyter-gpu


# Install Python 3.8 (compatible with ROS Noetic)
RUN apt-get update && apt-get install -y python3.8 python3.8-dev python3-pip && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.8 1

# Add ROS Noetic repository and install ROS from official sources
RUN apt-get update && apt-get install -y curl gnupg2 lsb-release && \
    sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list' && \
    curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | apt-key add - && \
    apt-get update && apt-get install -y ros-noetic-desktop-full && \
    echo "source /opt/ros/noetic/setup.bash" >> ~/.bashrc && \
    apt-get install -y python3-rosdep python3-rosinstall python3-rosinstall-generator python3-wstool build-essential && \
    rm -rf /var/lib/apt/lists/*

# Install any additional Python dependencies
COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt --ignore-installed


# Copy your main detection project
COPY . /detection

# Copy your ROS package
COPY ./ros_tools/road_lines_msg /catkin_ws/src/road_lines_msg

# Build the ROS workspace
RUN /bin/bash -c "source /opt/ros/noetic/setup.bash && \
    cd /catkin_ws && catkin_make && catkin_make install && \
    echo 'source /catkin_ws/devel/setup.bash' >> /root/.bashrc"

# Add entrypoint script
COPY ./ros_entrypoint.sh /ros_entrypoint.sh
RUN chmod +x /ros_entrypoint.sh

# Set working directory
WORKDIR /detection

# Set the entrypoint
ENTRYPOINT ["/ros_entrypoint.sh"]

# Default command

# Final fix: enforce numpy 1.19.5 to avoid ABI mismatch with cv_bridge
#RUN pip install --force-reinstall numpy==1.19.5

CMD ["python", "main.py"]
