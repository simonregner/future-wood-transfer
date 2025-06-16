import math
import numpy as np
import rclpy
from rclpy.node import Node
import tf_transformations  # Note: install 'tf_transformations' Python package!
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path

class SinglePathPublisher(Node):
    def __init__(self, topic_name='/pose_path'):
        super().__init__('single_path_publisher')
        self.publisher = self.create_publisher(Path, topic_name, 1)
        self.frame_id = None

    def publish_path(self, points, frame_id):
        if len(points) < 2:
            path = Path()
            path.header.stamp = self.get_clock().now().to_msg()
            path.header.frame_id = frame_id
            self.publisher.publish(path)
            return

        self.frame_id = frame_id

        path = Path()
        path.header.stamp = self.get_clock().now().to_msg()
        path.header.frame_id = self.frame_id

        poses = []
        for i in range(len(points) - 1):
            current_point = points[i]
            next_point = points[i + 1]

            dx = next_point[0] - current_point[0]
            dy = next_point[1] - current_point[1]
            dz = next_point[2] - current_point[2]

            yaw = math.atan2(dy, dx)
            distance_xy = math.sqrt(dx ** 2 + dy ** 2)
            pitch = math.atan2(-dz, distance_xy)  # Negative dz for ROS

            # Convert to quaternion: (roll, pitch, yaw)
            quaternion = tf_transformations.quaternion_from_euler(0, pitch, yaw)

            pose_stamped = PoseStamped()
            pose_stamped.header.stamp = self.get_clock().now().to_msg()
            pose_stamped.header.frame_id = self.frame_id
            pose_stamped.pose.position.x = current_point[0]
            pose_stamped.pose.position.y = current_point[1]
            pose_stamped.pose.position.z = current_point[2] if len(current_point) > 2 else 0.0
            pose_stamped.pose.orientation.x = quaternion[0]
            pose_stamped.pose.orientation.y = quaternion[1]
            pose_stamped.pose.orientation.z = quaternion[2]
            pose_stamped.pose.orientation.w = quaternion[3]

            poses.append(pose_stamped)

        path.poses = poses
        self.publisher.publish(path)