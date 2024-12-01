import rospy
from geometry_msgs.msg import PoseStamped, Quaternion
from nav_msgs.msg import Path
import tf.transformations as tf_transformations
import numpy as np

import math
import tf


def compute_orientation(point_a, point_b):
    direction = np.array(point_b) - np.array(point_a)
    yaw = np.arctan2(direction[1], direction[0])
    quaternion = tf_transformations.quaternion_from_euler(0, 0, yaw)
    return Quaternion(x=quaternion[0], y=quaternion[1], z=quaternion[2], w=quaternion[3])


class PathPublisher:
    def __init__(self, node_name='path_node', topic_name='/pose_path', frame_id='map'):
        #rospy.init_node(node_name, anonymous=True)
        self.publisher = rospy.Publisher(topic_name, Path, queue_size=10)
        self.frame_id = frame_id

    def publish_path(self, points):
        if len(points) < 2:
            rospy.logwarn("At least two points are required to create a path.")
            return

        path = Path()
        path.header.stamp = rospy.Time.now()
        path.header.frame_id = self.frame_id

        for i in range(len(points) - 1):
            current_point = points[i]
            next_point = points[i + 1]
            dx = next_point[0] - current_point[0]
            dy = next_point[1] - current_point[1]
            yaw = math.atan2(dy, dx)
            quaternion = tf.transformations.quaternion_from_euler(0, 0, yaw)

            pose_stamped = PoseStamped()
            pose_stamped.header.stamp = rospy.Time.now()
            pose_stamped.header.frame_id = self.frame_id
            pose_stamped.pose.position.x = current_point[0]
            pose_stamped.pose.position.y = current_point[1]
            pose_stamped.pose.position.z = current_point[2] if len(current_point) > 2 else 0.0
            pose_stamped.pose.orientation.x = quaternion[0]
            pose_stamped.pose.orientation.y = quaternion[1]
            pose_stamped.pose.orientation.z = quaternion[2]
            pose_stamped.pose.orientation.w = quaternion[3]

            path.poses.append(pose_stamped)

        self.publisher.publish(path)
        rospy.loginfo("Published path with {} poses.".format(len(path.poses)))

