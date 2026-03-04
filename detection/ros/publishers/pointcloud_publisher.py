import numpy as np
import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge
from sensor_msgs_py import point_cloud2
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header
import struct

class PointcloudPublisher(Node):
    def __init__(self, topic_name='/ml/pointcloud'):
        super().__init__('pointcloud_publisher')
        self.publisher = self.create_publisher(PointCloud2, topic_name, 5)
        self.publisher_right = self.create_publisher(PointCloud2, '/ml/pointcloud_right', 5)
        self.publisher_left = self.create_publisher(PointCloud2, '/ml/pointcloud_left', 5)
        self.bridge = CvBridge()
        self.frame_id = None

    def publish_pointcloud(self, points, pointcloud_right, pointcloud_left, frame_id):
        # The incoming points arrays must have shape Nx3 (for x,y,z), or Nx4 (if already RGB packed).
        rotated_points_right = pointcloud_right
        rotated_points_left = pointcloud_left
        rotated_points = points

        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
        ]
        fields_side = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='rgb', offset=12, datatype=PointField.FLOAT32, count=1),
        ]

        def rgb_to_float(r, g, b):
            rgb_int = (int(r) << 16) | (int(g) << 8) | int(b)
            return struct.unpack('f', struct.pack('I', rgb_int))[0]

        pc2_data = [tuple(p) for p in rotated_points]
        pc2_data_right = [tuple(list(p) + [rgb_to_float(255, 0, 0)]) for p in rotated_points_right]
        pc2_data_left = [tuple(list(p) + [rgb_to_float(0, 255, 0)]) for p in rotated_points_left]

        # Create Header with ROS2 style timestamp
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = frame_id

        pc2_msg = point_cloud2.create_cloud(header, fields, pc2_data)
        pc2_msg_right = point_cloud2.create_cloud(header, fields_side, pc2_data_right)
        pc2_msg_left = point_cloud2.create_cloud(header, fields_side, pc2_data_left)

        self.publisher.publish(pc2_msg)
        self.publisher_right.publish(pc2_msg_right)
        self.publisher_left.publish(pc2_msg_left)

# Example usage (outside the class):
# rclpy.init()
# node = PointcloudPublisher()
# node.publish_pointcloud(points, points_right, points_left, "map")
# rclpy.spin(node)
# node.destroy_node()
# rclpy.shutdown()
