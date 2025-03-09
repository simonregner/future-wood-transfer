cmake_minimum_required(VERSION 3.0.2)
project(road_lines_msg)

find_package(catkin REQUIRED COMPONENTS
    roscpp
    rospy
    std_msgs
    nav_msgs
    message_generation
)

add_message_files(
    FILES
    RoadLinesMsg.msg
)

generate_messages(
    DEPENDENCIES
    std_msgs
    nav_msgs
)

catkin_package(
    CATKIN_DEPENDS message_runtime std_msgs nav_msgs
)

include_directories(
    ${catkin_INCLUDE_DIRS}
)

add_executable(road_lines_subscriber src/road_lines_subscriber.cpp)
target_link_libraries(road_lines_subscriber ${catkin_LIBRARIES})
add_dependencies(road_lines_subscriber ${PROJECT_NAME}_generate_messages_cpp)
