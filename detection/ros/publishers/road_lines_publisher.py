import math
import rclpy
from rclpy.node import Node

from road_lines_msg.msg import RoadLinesMsg, RoadLine
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path

import tf_transformations  # pip install tf-transformations

class RoadLinesPublisher(Node):
    def __init__(self, topic_name='/path_array'):
        super().__init__('road_lines_publisher')
        self.publisher = self.create_publisher(RoadLinesMsg, topic_name, 10)
        self.frame_id = None

    def publish_path(self, point_array_left, point_array_right, frame_id, time_stamp):
        self.frame_id = frame_id

        msg = RoadLinesMsg()
        msg.header.stamp = time_stamp
        msg.header.frame_id = self.frame_id

        for i in range(len(point_array_left)):
            points_left = point_array_left[i]
            points_right = point_array_right[i]

            def create_path(points):
                path = Path()
                path.header.stamp = time_stamp
                path.header.frame_id = self.frame_id

                if len(points) == 0:
                    return path

                for i in range(len(points) - 1):
                    current_point = points[i]
                    next_point = points[i + 1]

                    dx = next_point[0] - current_point[0]
                    dy = next_point[1] - current_point[1]
                    dz = next_point[2] - current_point[2]

                    yaw = math.atan2(dy, dx)
                    distance_xy = math.sqrt(dx ** 2 + dy ** 2)
                    pitch = math.atan2(-dz, distance_xy)

                    quaternion = tf_transformations.quaternion_from_euler(0, pitch, yaw)

                    pose_stamped = PoseStamped()
                    pose_stamped.header.stamp = time_stamp
                    pose_stamped.header.frame_id = self.frame_id
                    pose_stamped.pose.position.x = current_point[0]
                    pose_stamped.pose.position.y = current_point[1]
                    pose_stamped.pose.position.z = current_point[2] if len(current_point) > 2 else 0.0
                    pose_stamped.pose.orientation.x = quaternion[0]
                    pose_stamped.pose.orientation.y = quaternion[1]
                    pose_stamped.pose.orientation.z = quaternion[2]
                    pose_stamped.pose.orientation.w = quaternion[3]

                    path.poses.append(pose_stamped)

                return path

            road_line = RoadLine()
            road_line.left_path = create_path(points_left)
            road_line.right_path = create_path(points_right)
            msg.paths.append(road_line)

        self.publisher.publish(msg)

# Example usage (for ROS2, in your main):
# rclpy.init()
# node = RoadLinesPublisher()
# node.publish_path(point_array_left, point_array_right, "map", node.get_clock().now().to_msg())
# rclpy.spin(node)
# node.destroy_node()
# rclpy.shutdown()
