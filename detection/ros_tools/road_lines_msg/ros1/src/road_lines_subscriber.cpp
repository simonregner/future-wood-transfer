#include <ros/ros.h>
#include <road_lines_msg/RoadLinesMsg.h>

void roadLinesCallback(const road_lines_msg::RoadLinesMsg::ConstPtr& msg) {
    ROS_INFO("Received RoadLinesMsg with %lu paths", msg->paths.size());
    for (size_t i = 0; i < msg->paths.size(); i++) {
        ROS_INFO("Path %lu has %lu poses", i, msg->paths[i].poses.size());
    }
}

int main(int argc, char** argv) {
    ros::init(argc, argv, "road_lines_subscriber");
    ros::NodeHandle nh;
    ros::Subscriber sub = nh.subscribe("road_lines_topic", 10, roadLinesCallback);
    ros::spin();
    return 0;
}
