# Future-Wood-Transfer

## Install with Docker 

### Build docker image

```
cd path/to/folder
```

```
sudo docker build -t future_wood_transfer .
```

### Download docker image from docker hub

```
sudo docker run --gpus all -it --rm --network host simonregner/future_wood_transfer:v1.0
```

```
install nvidia-container-toolkit
```

### Run docker image

```
sudo docker run --gpus all -it --rm --network host future_wood_transfer
```

## Conda install requirements

### Install mamba
Need to install mamba for ros

```
conda install mamba
mamba init
```

### Install ROS Noetic

https://robostack.github.io/GettingStarted.html

#### Prepare an environment to use the correct channels
```
mamba create -n fwt python=3.11

mamba activate fwt

conda config --env --add channels conda-forge

conda config --env --add channels robostack-staging

conda config --env --remove channels defaults
```

#### Install ROS1
```
mamba install ros-noetic-desktop
```

#### Install tools for local development
```
mamba install compilers cmake pkg-config make ninja colcon-common-extensions catkin_tools rosdep
```

### Install yolo (ultralytics)
```
mamba install -c conda-forge ultralytics
```

### Install pytorch
```
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
```

### Install other requirements 
```
mamba install open3d scikit-learn
```

# Run the code with bagfile
## Run ROS and bagfile
### First terminal 
```
roscore
```

### Second terminal 
```
rosbag play bagfile --loop
```

## Run python code 

### Third terminal 
```
python road_detection.py
```

### Docker Container

# Push Docker
```
sudo docker image tag future_wood_transfer:latest simonregner/future_wood_transfer:v1.5
```

```
sudo docker image push simonregner/future_wood_transfer:v1.5
```
