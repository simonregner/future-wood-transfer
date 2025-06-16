FROM ros:noetic-ros-base

# 1. System setup
RUN apt-get update && apt-get install -y \
    python3-catkin-tools \
    && rm -rf /var/lib/apt/lists/*

# 2. Create catkin workspace
ENV CATKIN_WS=/catkin_ws
RUN mkdir -p $CATKIN_WS/src

# 3. Copy your custom message package into the workspace
COPY ros1 $CATKIN_WS/src/road_lines_msg

# 4. Build the workspace
RUN /bin/bash -c "source /opt/ros/noetic/setup.bash && \
    cd $CATKIN_WS && \
    catkin_make"

# 5. Source overlay by default
RUN echo "source /opt/ros/noetic/setup.bash" >> /root/.bashrc && \
    echo "source /catkin_ws/devel/setup.bash" >> /root/.bashrc

WORKDIR $CATKIN_WS

CMD ["/bin/bash"]