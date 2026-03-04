#include <ros/ros.h>
#include <path_array_pkg/PathArray.h>

void pathArrayCallback(const path_array_pkg::PathArray::ConstPtr& msg) {
    ROS_INFO("Received PathArray with %lu paths", msg->paths.size());
    for (size_t i = 0; i < msg->paths.size(); i++) {
        ROS_INFO("Path %lu has %lu poses", i, msg->paths[i].poses.size());
    }
}

int main(int argc, char** argv) {
    ros::init(argc, argv, "path_array_subscriber");
    ros::NodeHandle nh;
    ros::Subscriber sub = nh.subscribe("path_array_topic", 10, pathArrayCallback);
    ros::spin();
    return 0;
}
