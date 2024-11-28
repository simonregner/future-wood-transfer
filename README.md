# Future-Wood-Transfer

## Conda install requirements

### Install mamba
Need to install mamba for ros

### Install ROS Noetic

https://robostack.github.io/GettingStarted.html

#### Prepare an environment to use the correct channels
mamba create -n fwt python=3.11

mamba activate fwt

conda config --env --add channels conda-forge

conda config --env --add channels robostack-staging

conda config --env --remove channels defaults#

#### Install ROS1
mamba install ros-noetic-desktop

#### Install tools for local development
mamba install compilers cmake pkg-config make ninja colcon-common-extensions catkin_tools rosdep

### Install yolo (ultralytics)
mamba install -c conda-forge ultralytics

### Install pytorch

mamba install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia

### Install other requirements 

mamba install open3d scikit-learn