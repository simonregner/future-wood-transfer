import math

import numpy as np
import rospy
import tf
import tf.transformations as tf_transformations
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path


class SinglePathPublisher:
    def __init__(self, topic_name='/pose_path'):
        self.publisher = rospy.Publisher(topic_name, Path, queue_size=1)
        self.frame_id = None

    def publish_path(self, points, frame_id):

        if len(points) < 2:
            #rospy.logwarn("At least two points are required to create a path.")
            path = Path()
            path.header.stamp = rospy.Time.now()
            path.header.frame_id = self.frame_id
            self.publisher.publish(path)
            return

        self.frame_id = frame_id



        # Initialize the Path message
        path = Path()
        path.header.stamp = rospy.Time.now()
        path.header.frame_id = self.frame_id

        # Compute yaw angles and build poses
        poses = []
        for i in range(len(points) - 1):
            current_point = points[i]
            next_point = points[i + 1]
            # current_point and next_point should be [x, y, z]
            dx = next_point[0] - current_point[0]
            dy = next_point[1] - current_point[1]
            dz = next_point[2] - current_point[2]

            # Compute yaw (rotation around z) and pitch (rotation around y)
            yaw = math.atan2(dy, dx)
            distance_xy = math.sqrt(dx ** 2 + dy ** 2)
            pitch = math.atan2(-dz, distance_xy)  # Negative dz for typical ROS coordinate convention

            # Convert to quaternion: (roll, pitch, yaw)
            quaternion = tf.transformations.quaternion_from_euler(0, pitch, yaw)

            # Create PoseStamped message
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

            poses.append(pose_stamped)

        path.poses = poses  # Assign all poses at once for efficiency
        self.publisher.publish(path)
