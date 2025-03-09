import numpy as np
import rospy
from cv_bridge import CvBridge
from sensor_msgs import point_cloud2
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header


class PointcloudPublisher:
    def __init__(self, topic_name='/ml/pointcloud'):
        self.publisher = rospy.Publisher(topic_name, PointCloud2, queue_size=5)
        self.publisher_right = rospy.Publisher('/ml/pointcloud_right', PointCloud2, queue_size=5)
        self.publisher_left = rospy.Publisher('/ml/pointcloud_left', PointCloud2, queue_size=5)

        # OpenCV Bridge for converting images to ROS messages
        self.bridge = CvBridge()

        self.frame_id = None

    def publish_pointcloud(self, points, pointcloud_right, pointcloud_left, frame_id):

        rotated_points_right = pointcloud_right  # @ rotation_matrix_open3d_to_ros.T
        rotated_points_left = pointcloud_left  # @ rotation_matrix_open3d_to_ros.T

        rotated_points = points  # @ rotation_matrix_open3d_to_ros.T

        # Create PointCloud2 message fields
        fields = [
            PointField('x', 0, PointField.FLOAT32, 1),
            PointField('y', 4, PointField.FLOAT32, 1),
            PointField('z', 8, PointField.FLOAT32, 1)
        ]

        # Convert rotated_points to a list of tuples
        pc2_data = [tuple(p) for p in rotated_points]
        pc2_data_right = [tuple(p) for p in rotated_points_right]
        pc2_data_left = [tuple(p) for p in rotated_points_left]

        # Create initial PointCloud2
        header = Header(frame_id=frame_id)
        pc2_msg = point_cloud2.create_cloud(header, fields, pc2_data)
        pc2_msg_right = point_cloud2.create_cloud(header, fields, pc2_data_right)
        pc2_msg_left = point_cloud2.create_cloud(header, fields, pc2_data_left)

        self.publisher.publish(pc2_msg)
        self.publisher_right.publish(pc2_msg_right)
        self.publisher_left.publish(pc2_msg_left)
