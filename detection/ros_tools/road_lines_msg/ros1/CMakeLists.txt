#cmake_minimum_required(VERSION 3.5)
#project(road_lines_msg)

#find_package(ament_cmake REQUIRED)
#find_package(geometry_msgs REQUIRED)
#find_package(std_msgs REQUIRED)
#find_package(nav_msgs REQUIRED)
#find_package(sensor_msgs REQUIRED)
#find_package(rosidl_default_generators REQUIRED)

#rosidl_generate_interfaces(${PROJECT_NAME}
#  "msg/RoadLinesMsg.msg"
#  "msg/RoadLine.msg"
#  DEPENDENCIES geometry_msgs std_msgs nav_msgs sensor_msgs
#)

#ament_package()




cmake_minimum_required(VERSION 3.5)
project(road_lines_msg)

## Compile as C++11, supported in ROS Kinetic and newer
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  message_generation
  message_runtime
  roscpp
  rospy
  nav_msgs
  std_msgs
  geometry_msgs
  sensor_msgs
)



# Message file
add_message_files(
  DIRECTORY
    msg
  FILES
    RoadLine.msg
    RoadLinesMsg.msg
)

generate_messages(
  DEPENDENCIES
    nav_msgs
    std_msgs
    geometry_msgs
    sensor_msgs
)

catkin_package(
  CATKIN_DEPENDS
    message_runtime
)


#catkin_install_python(
#  PROGRAMS
#    src/road_lines_publisher.py
#    src/road_lines_subscriber.py
#  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
#)