cmake_minimum_required(VERSION 3.0.2)
project(road_lines_msg)


find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  message_generation
  roscpp
  rospy
  std_msgs
  nav_msgs
  sensor_msgs
)

add_message_files(
  FILES
  RoadLinesMsg.msg
  RoadLine.msg
)

generate_messages(DEPENDENCIES	
  geometry_msgs
  nav_msgs
  sensor_msgs
  std_msgs
)


catkin_package(
  CATKIN_DEPENDS message_runtime std_msgs nav_msgs
)

include_directories(include ${catkin_INCLUDE_DIRS})

# target_link_libraries(road_lines_msg ${catkin_LIBRARIES})
# add_dependencies(road_lines_msg ${catkin_EXPORTED_TARGETS})