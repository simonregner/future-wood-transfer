import rospy

from geometry_msgs.msg import PoseStamped, Quaternion
from std_msgs.msg import Float64MultiArray
from nav_msgs.msg import Path

import numpy as np
import math

import tf
import tf.transformations as tf_transformations


class PathPublisher:
    def __init__(self, topic_name='/path_array'):
        self.publisher = rospy.Publisher(topic_name, Float64MultiArray, queue_size=1)
        self.frame_id = None


    def publish_path(self, point_array, frame_id):
        self.frame_id = frame_id

        path_array = []

        for point_pair in point_array:
            path_pair = []
            for i, points in enumerate(point_pair):
                # Initialize the Path message
                path = Path()
                path.header.stamp = rospy.Time.now()
                path.header.frame_id = self.frame_id

                # Compute yaw angles and build poses
                poses = []
                for i in range(2, len(points) - 1):
                    current_point = points[i]
                    next_point = points[i + 1]
                    dx, dy = next_point[0] - current_point[0], next_point[1] - current_point[1]
                    yaw = math.atan2(dy, dx)
                    quaternion = tf.transformations.quaternion_from_euler(0, 0, yaw)

                    # Create PoseStamped message
                    pose_stamped = PoseStamped()
                    pose_stamped.header.stamp = rospy.Time.now()
                    pose_stamped.header.frame_id = self.frame_id
                    pose_stamped.pose.position.x = current_point[0]
                    pose_stamped.pose.position.y = current_point[1]
                    pose_stamped.pose.position.z = current_point[2] if current_point.shape[0] > 2 else 0.0
                    pose_stamped.pose.orientation.x = quaternion[0]
                    pose_stamped.pose.orientation.y = quaternion[1]
                    pose_stamped.pose.orientation.z = quaternion[2]
                    pose_stamped.pose.orientation.w = quaternion[3]

                    poses.append(pose_stamped)

                path.poses = poses  # Assign all poses at once for efficiency
                path_pair.append(path)

            path_array.append(path_pair)

        msg = Float64MultiArray()
        msg.data = path_array

        self.publisher.publish(msg)

class SinglePathPublisher:
    def __init__(self, topic_name='/pose_path'):
        self.publisher = rospy.Publisher(topic_name, Path, queue_size=1)
        self.frame_id = None

    def publish_path(self, points, frame_id):

        if len(points) < 2:
            rospy.logwarn("At least two points are required to create a path.")
            return

        self.frame_id = frame_id

        # Define a rotation matrix to align to ROS convention
        # 90-degree rotation around X-axis
        rotation_matrix_open3d_to_ros = np.array([
            [0, 0, 1],  # Z-axis of Open3D becomes X-axis of ROS
            [-1, 0, 0],  # -X-axis of Open3D becomes Y-axis of ROS
            [0, 1, 0]  # Y-axis of Open3D becomes Z-axis of ROS
        ])

        # Apply the rotation
        points_ros = points #@ rotation_matrix_open3d_to_ros.T

        # Initialize the Path message
        path = Path()
        path.header.stamp = rospy.Time.now()
        path.header.frame_id = self.frame_id

        # Compute yaw angles and build poses
        poses = []
        for i in range(2, len(points_ros) - 1):
            current_point = points_ros[i]
            next_point = points_ros[i + 1]
            dx, dy = next_point[0] - current_point[0], next_point[1] - current_point[1]
            yaw = math.atan2(dy, dx)
            quaternion = tf.transformations.quaternion_from_euler(0, 0, yaw)

            # Create PoseStamped message
            pose_stamped = PoseStamped()
            pose_stamped.header.stamp = rospy.Time.now()
            pose_stamped.header.frame_id = self.frame_id
            pose_stamped.pose.position.x = current_point[0]
            pose_stamped.pose.position.y = current_point[1]
            pose_stamped.pose.position.z = current_point[2] if current_point.shape[0] > 2 else 0.0
            pose_stamped.pose.orientation.x = quaternion[0]
            pose_stamped.pose.orientation.y = quaternion[1]
            pose_stamped.pose.orientation.z = quaternion[2]
            pose_stamped.pose.orientation.w = quaternion[3]

            poses.append(pose_stamped)

        path.poses = poses  # Assign all poses at once for efficiency
        self.publisher.publish(path)