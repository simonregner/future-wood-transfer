FROM ros:humble-ros-base-jammy
# The above base image is multi-platform (works on ARM64 and AMD64):
# Docker will automatically select the correct platform variant based on the host's architecture.

#
# How to build this docker image:
#  docker build . -t ros-humble-ros1-bridge-builder --network host
#
# How to build ros-humble-ros1-bridge:
#  # 0.) From a Ubuntu 22.04 (Jammy) ROS 2 Humble system, create a "ros-humble-ros1-bridge/" ROS2 package:
#    docker run --network host --rm ros-humble-ros1-bridge-builder | tar xvzf -
#
# How to use the ros-humble-ros1-bridge:
#  # 1.) First start a ROS1 Noetic docker and bring up a GUI terminal, something like:
#    rocker --x11 --user --privileged --persist-image \
#         --volume /dev/shm /dev/shm --network=host -- ros:noetic-ros-base-focal \
#         'bash -c "sudo apt update; sudo apt install -y ros-noetic-rospy-tutorials tilix; tilix"'
#
#  # 2.) Then, start "roscore" inside the ROS1 container:
#    source /opt/ros/noetic/setup.bash
#    roscore
#
#  # 3.) Now, from the Ubuntu 22.04 (Jammy) ROS2 Desktop Humble system, start the ros1 bridge node:
#    apt-get -y install ros-humble-desktop
#    source /opt/ros/humble/setup.bash
#    source ros-humble-ros1-bridge/install/local_setup.bash
#    ros2 run ros1_bridge dynamic_bridge
#
#  # 4.) Back to the ROS1 Noetic container, run in another terminal tab:
#    source /opt/ros/noetic/setup.bash
#    rosrun rospy_tutorials talker
#
#  # 5.) Finally, from the Ubuntu 22.04 (Jammy) ROS2 Humble system:
#    source /opt/ros/humble/setup.bash
#    ros2 run demo_nodes_cpp listener
#

# Make sure bash catches errors (no need to chain commands with &&, use ; instead)
SHELL ["/bin/bash", "-o", "pipefail", "-o", "errexit", "-c"]


###########################
# 1.) Bring system up to the latest ROS desktop configuration
###########################
RUN apt-get -y update
RUN apt-get -y upgrade
RUN apt-get -y install ros-humble-desktop

###########################
# 2.) Temporarily remove ROS2 apt repository
###########################
RUN mv /etc/apt/sources.list.d/ros2.sources /root/
RUN apt-get update

###########################
# 3.) comment out the catkin conflict
###########################
RUN sed  -i -e 's|^Conflicts: catkin|#Conflicts: catkin|' /var/lib/dpkg/status
RUN apt-get install -f

###########################
# 4.) force install these packages
###########################
RUN apt-get download python3-catkin-pkg
RUN apt-get download python3-rospkg
RUN apt-get download python3-rosdistro
RUN dpkg --force-overwrite -i python3-catkin-pkg*.deb
RUN dpkg --force-overwrite -i python3-rospkg*.deb
RUN dpkg --force-overwrite -i python3-rosdistro*.deb
RUN apt-get install -f


###########################
# 5.) Install the latest ROS1 desktop configuration
# see https://packages.ubuntu.com/jammy/ros-desktop-dev
# note: ros-desktop-dev automatically includes tf tf2
###########################
RUN apt-get -y install ros-desktop-dev

# fix ARM64 pkgconfig path issue -- Fix provided by ambrosekwok
RUN if [[ $(uname -m) = "arm64" || $(uname -m) = "aarch64" ]]; then                     \
      cp /usr/lib/x86_64-linux-gnu/pkgconfig/* /usr/lib/aarch64-linux-gnu/pkgconfig/;   \
    fi

###########################
# 6.) Restore the ROS2 apt repos and set compilation options.
#   For example, to include ROS tutorial message types, pass
#   "--build-arg ADD_ros_tutorials=1" to the docker build command.
###########################
RUN mv /root/ros2.sources /etc/apt/sources.list.d/
RUN apt-get -y update

# for ros-humble-example-interfaces:
ARG ADD_ros_tutorials=1

# for ros-humble-grid-map:
ARG ADD_grid_map=0

# for a custom message example
ARG ADD_example_custom_msgs=1

# for octomap
ARG ADD_octomap_msgs=0


# Arguments to enable building your custom msg overlay, if you want to make it optional:
ARG ADD_road_lines_msg=1

# sanity check:
RUN echo "ADD_ros_tutorials         = '$ADD_ros_tutorials'"
RUN echo "ADD_grid_map              = '$ADD_grid_map'"
RUN echo "ADD_example_custom_msgs   = '$ADD_example_custom_msgs'"
RUN echo "ADD_octomap_msgs          = '$ADD_octomap_msgs'"
RUN echo "ADD_road_lines_msg        = '$ADD_road_lines_msg'"

###########################
# 6.1) Add ROS1 ros_tutorials messages and services
# eg., See AddTwoInts server and client tutorial
###########################
RUN if [[ "$ADD_ros_tutorials" = "1" ]]; then                                           \
      git clone -b noetic-devel --depth=1 https://github.com/ros/ros_tutorials.git;     \
      cd ros_tutorials;                                                                 \
      unset ROS_DISTRO;                                                                 \
      time colcon build --cmake-args -DCMAKE_BUILD_TYPE=Release;                        \
    fi

# unit test (optional)
# RUN if [[ "$ADD_ros_tutorials" = "1" ]]; then           \
#       cd ros_tutorials;                                 \
#       unset ROS_DISTRO;                                 \
#       colcon test --event-handlers console_direct+;     \
#       colcon test-result;                               \
#     fi

###########################
# 6.2 Add ROS1 grid-map messages
###########################
# navigation stuff (just need costmap_2d?)
RUN if [[ "$ADD_grid_map" = "1" ]]; then                                                        \
      apt-get -y install libsdl1.2-dev libsdl-image1.2-dev;                                     \
      git clone -b noetic-devel --depth=1 https://github.com/ros-planning/navigation.git;       \
      cd navigation;                                                                            \
      unset ROS_DISTRO;                                                                         \
      time colcon build --cmake-args -DCMAKE_BUILD_TYPE=Release                                 \
        --packages-select map_server voxel_grid costmap_2d;                                     \
    fi

# filter stuff
RUN if [[ "$ADD_grid_map" = "1" ]]; then                                        \
      git clone -b noetic-devel --depth=1 https://github.com/ros/filters.git;   \
      cd filters;                                                               \
      unset ROS_DISTRO;                                                         \
      time colcon build --cmake-args -DCMAKE_BUILD_TYPE=Release;                \
    fi

# finally grid-amp (only select a subset for now)
RUN if [[ "$ADD_grid_map" = "1" ]]; then                                                \
      apt-get -y install libpcl-ros-dev libcv-bridge-dev;                               \
      source navigation/install/setup.bash;                                             \
      source filters/install/setup.bash;                                                \
      git clone -b 1.6.4 --depth=1 https://github.com/ANYbotics/grid_map.git;   \
      cd grid_map;                                                                      \
      unset ROS_DISTRO;                                                                 \
      grep -r c++11 | grep CMakeLists | cut -f 1 -d ':' |                               \
        xargs sed -i -e 's|std=c++11|std=c++17|g';                                      \
      time colcon build --cmake-args -DCMAKE_BUILD_TYPE=Release                         \
        --packages-select grid_map_msgs grid_map_core grid_map_octomap grid_map_sdf     \
        grid_map_costmap_2d grid_map_cv grid_map_ros grid_map_loader;                   \
    fi

######################################
# 6.3) Compile custom message (code provided by Codaero)
#   Note1: Make sure the package name ends with "_msgs".
#   Note2: Use the same package name for both ROS1 and ROS2.
#   See https://github.com/ros2/ros1_bridge/blob/master/doc/index.rst
######################################
RUN if [[ "$ADD_example_custom_msgs" = "1" ]]; then                     \
      git clone https://github.com/TommyChangUMD/custom_msgs.git;       \
      #                                                                 \
      # Compile for ROS1:                                               \
      #                                                                 \
      cd /custom_msgs/custom_msgs_ros1;                                 \
      unset ROS_DISTRO;                                                 \
      time colcon build --cmake-args -DCMAKE_BUILD_TYPE=Release;        \
      #                                                                 \
      # Compile for ROS2:                                               \
      #                                                                 \
      cd /custom_msgs/custom_msgs_ros2;                                 \
      source /opt/ros/humble/setup.bash;                                \
      time colcon build --cmake-args -DCMAKE_BUILD_TYPE=Release;        \
    fi


###########################
# 6.4 Add ROS1 octomap message
###########################
RUN if [[ "$ADD_octomap_msgs" = "1" ]]; then                                    \
    git clone --depth 1 -b 0.3.5 https://github.com/OctoMap/octomap_msgs.git;   \
    cd octomap_msgs/;                                                           \
    unset ROS_DISTRO;                                                           \
    time colcon build --cmake-args -DCMAKE_BUILD_TYPE=Release;                  \
fi

#############################################
# 6.5) Compile road_lines_msg for ROS1 and ROS2
#############################################
COPY road_lines_msg/ros1 /road_lines_msg/road_lines_msg_ros1
COPY road_lines_msg/ros2 /road_lines_msg/road_lines_msg_ros2

RUN apt-get update && apt-get install -y ros-humble-ament-cmake

RUN if [[ "$ADD_road_lines_msg" = "1" ]]; then \
      cd /road_lines_msg/road_lines_msg_ros1; \
      unset ROS_DISTRO; \
      time colcon build --cmake-args -DCMAKE_BUILD_TYPE=Release; \
    fi

RUN if [[ "$ADD_road_lines_msg" = "1" ]]; then \
      cd /road_lines_msg/road_lines_msg_ros2; \
      source /opt/ros/humble/setup.bash; \
      time colcon build --cmake-args -DCMAKE_BUILD_TYPE=Release; \
    fi



###########################
# 7.) Compile ros1_bridge
###########################
RUN                                                                                             \
    #-------------------------------------                                                      \
    # Apply the ROS2 underlay                                                                   \
    #-------------------------------------                                                      \
    source /opt/ros/humble/setup.bash;                                                          \
    #                                                                                           \
    #-------------------------------------                                                      \
    # Apply additional message / service overlays                                               \
    #-------------------------------------                                                      \
    if [[ "$ADD_ros_tutorials" = "1" ]]; then                                                   \
      # Apply ROS1 package overlay                                                              \
      source ros_tutorials/install/setup.bash;                                                  \
      # Apply ROS2 package overlay                                                              \
      apt-get -y install ros-humble-example-interfaces;                                         \
      source /opt/ros/humble/setup.bash;                                                        \
    fi;                                                                                         \
    #                                                                                           \
    if [[ "$ADD_grid_map" = "1" ]]; then                                                        \
      # Apply ROS1 package overlay                                                              \
      source grid_map/install/setup.bash;                                                       \
      # Apply ROS2 package overlay                                                              \
      apt-get -y install ros-humble-grid-map;                                                   \
      source /opt/ros/humble/setup.bash;                                                        \
    fi;                                                                                         \
    #                                                                                           \
    if [[ "$ADD_example_custom_msgs" = "1" ]]; then                                             \
      # Apply ROS1 package overlay                                                              \
      source /custom_msgs/custom_msgs_ros1/install/setup.bash;                                  \
      # Apply ROS2 package overlay                                                              \
      source /custom_msgs/custom_msgs_ros2/install/setup.bash;                                  \
    fi;   \
                                                                                   \
    #                                                                                           \
    if [[ "$ADD_octomap_msgs" = "1" ]]; then                                                    \
      # Apply ROS1 package overlay                                                              \
      source octomap_msgs/install/setup.bash;                                                   \
      # Apply ROS2 package overlay                                                              \
      apt-get -y install ros-humble-octomap-msgs;                                               \
      source /opt/ros/humble/setup.bash;                                                        \
    fi;                                                                                         \
    #                                                                                           \
    if [[ "$ADD_road_lines_msg" = "1" ]]; then                                                  \
      source /road_lines_msg/road_lines_msg_ros1/install/setup.bash;                            \
      source /road_lines_msg/road_lines_msg_ros2/install/setup.bash;                            \
    fi;                                                                                         \
    #-------------------------------------                                                      \
    # Finally, build the Bridge                                                                 \
    #-------------------------------------                                                      \
    mkdir -p /ros-humble-ros1-bridge/src;                                                       \
    cd /ros-humble-ros1-bridge/src;                                                             \
    git clone -b action_bridge_humble --depth=1 https://github.com/smith-doug/ros1_bridge.git;  \
    cd ros1_bridge/;                                                                            \
    cd ../..;                                                                                   \
    MEMG=$(printf "%.0f" $(free -g | awk '/^Mem:/{print $2}'));                                 \
    NPROC=$(nproc);  MIN=$((MEMG<NPROC ? MEMG : NPROC));                                        \
    #                                                                                           \
    echo "Please wait...  running $MIN concurrent jobs to build ros1_bridge";                   \
    time MAKEFLAGS="-j $MIN" colcon build --event-handlers console_direct+                      \
      --cmake-args -DCMAKE_BUILD_TYPE=Release

###########################
# 8.) Clean up
###########################
RUN apt-get -y clean all; apt-get -y update

###########################
# 9.) Pack all ROS1 dependent libraries
###########################
# fix ARM64 pkgconfig path issue -- Fix provided by ambrosekwok
RUN if [[ $(uname -m) = "arm64" || $(uname -m) = "aarch64" ]]; then                    \
      cp /usr/lib/x86_64-linux-gnu/pkgconfig/* /usr/lib/aarch64-linux-gnu/pkgconfig/;  \
    fi

RUN ROS1_LIBS="libxmlrpcpp.so";                                                 \
     ROS1_LIBS="$ROS1_LIBS librostime.so";                                      \
     ROS1_LIBS="$ROS1_LIBS libroscpp.so";                                       \
     ROS1_LIBS="$ROS1_LIBS libroscpp_serialization.so";                         \
     ROS1_LIBS="$ROS1_LIBS librosconsole.so";                                   \
     ROS1_LIBS="$ROS1_LIBS librosconsole_log4cxx.so";                           \
     ROS1_LIBS="$ROS1_LIBS librosconsole_backend_interface.so";                 \
     ROS1_LIBS="$ROS1_LIBS liblog4cxx.so";                                      \
     ROS1_LIBS="$ROS1_LIBS libcpp_common.so";                                   \
     ROS1_LIBS="$ROS1_LIBS libb64.so";                                          \
     ROS1_LIBS="$ROS1_LIBS libaprutil-1.so";                                    \
     ROS1_LIBS="$ROS1_LIBS libapr-1.so";                                        \
     ROS1_LIBS="$ROS1_LIBS libactionlib.so.1d";                                 \
     cd /ros-humble-ros1-bridge/install/ros1_bridge/lib;                        \
     for soFile in $ROS1_LIBS; do                                               \
       soFilePath=$(ldd libros1_bridge.so | grep $soFile | awk '{print $3;}');  \
       cp $soFilePath ./;                                                       \
     done


###########################
# 10.) Start the dynamic bridge automatically
###########################
#
CMD ["/bin/bash"]
#
RUN printf '#!/bin/bash\n\
set -e\n\
source /opt/ros/humble/setup.bash\n\
source /ros-humble-ros1-bridge/install/setup.bash\n\
exec ros2 run ros1_bridge dynamic_bridge --bridge-all-topics\n' > /ros_entrypoint.sh \
  && chmod +x /ros_entrypoint.sh

ENTRYPOINT ["/ros_entrypoint.sh"]