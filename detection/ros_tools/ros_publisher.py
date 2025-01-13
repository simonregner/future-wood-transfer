import rospy
from geometry_msgs.msg import PoseStamped, Quaternion
from nav_msgs.msg import Path
from sensor_msgs.msg import Image, PointCloud2, PointField
from sensor_msgs import point_cloud2
import tf.transformations as tf_transformations
import numpy as np

import math
import tf

import cv2
from cv_bridge import CvBridge

import time

from std_msgs.msg import Header


def compute_orientation(point_a, point_b):
    direction = np.array(point_b) - np.array(point_a)
    yaw = np.arctan2(direction[1], direction[0])
    quaternion = tf_transformations.quaternion_from_euler(0, 0, yaw)
    return Quaternion(x=quaternion[0], y=quaternion[1], z=quaternion[2], w=quaternion[3])


class PathPublisher:
    def __init__(self, topic_name='/pose_path'):
        self.publisher = rospy.Publisher(topic_name, Path, queue_size=5)
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
        points_ros = points @ rotation_matrix_open3d_to_ros.T

        path = Path()
        path.header.stamp = rospy.Time.now()
        path.header.frame_id = self.frame_id

        for i in range(2, len(points_ros) - 1):
            current_point = points_ros[i]
            next_point = points_ros[i + 1]
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

class MaskPublisher:
    def __init__(self,  topic_name='/ml/mask_image'):
        self.publisher = rospy.Publisher(topic_name, Image, queue_size=5)

        # OpenCV Bridge for converting images to ROS messages
        self.bridge = CvBridge()

        self.frame_id = None


    def publish_mask(self, image, results, frame_id):
        if results is None:
            image_msg = self.bridge.cv2_to_imgmsg(image, encoding="bgr8")
            self.publisher.publish(image_msg)
            return

        self.frame_id = frame_id

        # Get the original image dimensions
        image_height, image_width = image.shape[:2]

        masks = results[0].masks
        # Create a blank image for the mask overlay
        mask_overlay = np.zeros_like(image)

        # Apply each mask to the overlay with a different color
        for i, mask in enumerate(masks.data):
            # Convert the mask to a binary numpy array
            binary_mask = mask.cpu().numpy().astype(np.uint8)

            scaled_mask = cv2.resize(binary_mask, (image_width, image_height), interpolation=cv2.INTER_NEAREST)

            # Create a random color for the mask
            color = [255, 0, 0]

            # Apply the color to the mask
            mask_colored = np.stack([scaled_mask * c for c in color], axis=-1)

            # Add the colored mask to the overlay
            mask_overlay = cv2.addWeighted(mask_overlay, 1.0, mask_colored, 0.5, 0)

        # Combine the mask overlay with the original image
        result_image = cv2.addWeighted(image, 1, mask_overlay, 0.75, 0)


        image_msg = self.bridge.cv2_to_imgmsg(result_image, encoding="bgr8")

        image_msg.header = Header()
        image_msg.header.stamp = rospy.Time.now()

        self.publisher.publish(image_msg)


class PointcloudPublisher:
    def __init__(self, topic_name='/ml/pointcloud'):
        self.publisher = rospy.Publisher(topic_name, PointCloud2, queue_size=5)
        self.publisher_right = rospy.Publisher('/ml/pointcloud_right', PointCloud2, queue_size=5)
        self.publisher_left = rospy.Publisher('/ml/pointcloud_left', PointCloud2, queue_size=5)


        # OpenCV Bridge for converting images to ROS messages
        self.bridge = CvBridge()

        self.frame_id = None

    def publish_pointcloud(self, points, pointcloud_right, pointcloud_left, frame_id):

        rotation_matrix_open3d_to_ros = np.array([
            [0, 0, 1],  # Z-axis of Open3D becomes X-axis of ROS
            [-1, 0, 0],  # -X-axis of Open3D becomes Y-axis of ROS
            [0, 1, 0]  # Y-axis of Open3D becomes Z-axis of ROS
        ])

        rotated_points_right = pointcloud_right @ rotation_matrix_open3d_to_ros.T
        rotated_points_left = pointcloud_left @ rotation_matrix_open3d_to_ros.T

        rotated_points = points @ rotation_matrix_open3d_to_ros.T

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


        # Create initial PointCloud2 (header timestamp will be updated when publishing)
        header = Header(frame_id=frame_id)
        pc2_msg = point_cloud2.create_cloud(header, fields, pc2_data)
        pc2_msg_right = point_cloud2.create_cloud(header, fields, pc2_data_right)
        pc2_msg_left = point_cloud2.create_cloud(header, fields, pc2_data_left)


        self.publisher.publish(pc2_msg)
        self.publisher_right.publish(pc2_msg_right)
        self.publisher_left.publish(pc2_msg_left)