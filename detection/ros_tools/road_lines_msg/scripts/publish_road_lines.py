#!/usr/bin/env python
import rospy
from road_lines_msg.msg import RoadLinesMsg
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped

def create_dummy_path():
    path = Path()
    path.header.stamp = rospy.Time.now()
    path.header.frame_id = "map"

    for i in range(5):
        pose = PoseStamped()
        pose.header = path.header
        pose.pose.position.x = i
        pose.pose.position.y = i * 0.5
        path.poses.append(pose)

    return path

def publisher():
    rospy.init_node("road_lines_publisher", anonymous=True)
    pub = rospy.Publisher("road_lines_topic", RoadLinesMsg, queue_size=10)
    rate = rospy.Rate(1)

    while not rospy.is_shutdown():
        msg = RoadLinesMsg()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = "map"
        msg.paths = [create_dummy_path(), create_dummy_path()]  # Example with two paths

        pub.publish(msg)
        rate.sleep()

if __name__ == "__main__":
    publisher()
