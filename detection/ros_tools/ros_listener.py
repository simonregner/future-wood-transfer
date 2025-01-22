import sys
sys.path.append("..")

import rospy
from sensor_msgs.msg import Image, CompressedImage, CameraInfo
from message_filters import Subscriber, TimeSynchronizer

from cv_bridge import CvBridge

import numpy as np

from detection.pointcloud.depth_to_pointcloud import depth_to_pointcloud_from_mask
from detection.pointcloud.pointcloud_edge_detection import edge_detection, edge_detection_2d
from detection.pointcloud.pointcloud_converter import pointcloud_to_2d

import detection.ros_tools.publisher.ros_path_publisher as ros_path_publisher
import detection.ros_tools.publisher.ros_mask_publisher as ros_mask_publisher
import detection.ros_tools.publisher.ros_pointcloud_publisher as ros_pointcloud_publisher

from detection.path.point_function import fit_line_3d_smooth, fit_polynomial

from detection.tools.mask import  keep_largest_component

import time

import cv2

from scipy.interpolate import splprep, splev

from scipy.interpolate import make_interp_spline

from scipy.signal import savgol_filter

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline

from rospy.timer import Rate

last_processed_time = rospy.Time()

class TimeSyncListener:
    def __init__(self, model_loader):
        # Initialize the node
        rospy.init_node('time_sync_listener', anonymous=True)

        # Store the model
        self.model_loader = model_loader

        self.intrinsic_matrix = None
        self.camera_info_topic = "/hazard_front/zed_node_front/depth/camera_info"

        self.bridge = CvBridge()

        # Dynamically check and assign the appropriate topic for image_sub
        self.rgb_image_type = None
        self.image_sub = None

        # Set timer for pause
        # TODO: Change for not static
        self.timers = [200000000, 200000000, 200000000, 200000000, 200000000]

        # Subscribers for the topics
        self.depth_sub = Subscriber('/hazard_front/zed_node_front/depth/depth_registered', Image)

        self.get_available_image_topic()

        # Synchronize the topics using TimeSynchronizer
        self.ts = TimeSynchronizer([self.image_sub, self.depth_sub], 1)
        self.ts.registerCallback(self.callback)

        self.left_path_publisher = ros_path_publisher.SinglePathPublisher(topic_name='/path/left_path')
        self.right_path_publisher = ros_path_publisher.SinglePathPublisher(topic_name='/path/right_path')

        self.path_publisher = ros_path_publisher.PathPublisher(topic_name='/path_publisher')

        self.mask_image_publisher = ros_mask_publisher.MaskPublisher(topic_name='/ml/mask_image')

        self.point_cloud_publisher = ros_pointcloud_publisher.PointcloudPublisher(topic_name='/ml/pointcloud')

        rospy.loginfo("Time sync listener initialized and running")

    def get_available_image_topic(self):
        """
        Checks if the compressed image topic exists and returns the appropriate topic.
        """
        available_topics = [topic for topic, _ in rospy.get_published_topics()]
        compressed_topic = "/hazard_front/zed_node_front/left/image_rect_color/compressed"
        uncompressed_topic = "/hazard_front/zed_node_front/left/image_rect_color"

        if compressed_topic in available_topics:
            rospy.loginfo(f"Topic found: {compressed_topic}")
            self.image_sub = Subscriber(compressed_topic, CompressedImage)
            self.rgb_image_type = CompressedImage
            return
        elif uncompressed_topic in available_topics:
            rospy.loginfo(f"Topic found: {uncompressed_topic}")
            self.image_sub = Subscriber(uncompressed_topic, Image)
            self.rgb_image_type = Image
            return
        else:
            rospy.logerr("Neither compressed nor uncompressed image topic is available!")
            rospy.signal_shutdown("No suitable image topic found.")
            return

    def single_listen(self, topic_name, message_type):
        """
        Listens to a topic once and processes the received message.

        :param topic_name: Name of the topic to listen to.
        :param message_type: Type of the ROS message expected.
        """
        rospy.loginfo(f"Listening to topic {topic_name} once.")

        try:
            # Wait for a single message on the topic
            msg = rospy.wait_for_message(topic_name, message_type, timeout=5)
            rospy.loginfo(f"Received a message from {topic_name}")

            # Example processing of the message (convert to OpenCV if it's an Image)
            if message_type == CameraInfo:
                self.intrinsic_matrix = np.array(msg.K).reshape(3, 3)


                print(self.intrinsic_matrix)
                print(msg)
        except rospy.ROSException as e:
            rospy.logerr(f"Failed to receive a message on {topic_name}: {e}")

    def callback(self, image_msg, depth_msg):
        """
        Callback function that handles synchronized messages.
        """
        global last_processed_time

        # Skip if still processing a previous frame
        if rospy.Time.now() - last_processed_time < rospy.Duration(0, int(sum(self.timers) / 5) + 50000000):
            return

        last_processed_time = rospy.Time.now()
        start_time = time.time()

        # Initialize intrinsic matrix and image type if not already done
        if self.intrinsic_matrix is None:
            self.single_listen(self.camera_info_topic, CameraInfo)

        if self.rgb_image_type is None:
            self.get_available_image_topic()
            return

        # Validate timestamps for synchronization
        if image_msg.header.stamp != depth_msg.header.stamp:
            rospy.loginfo("NO synchronized messages received")
            return

        frame_id = image_msg.header.frame_id

        # Convert RGB image
        try:
            if self.rgb_image_type is Image:
                rgb_image = self.bridge.imgmsg_to_cv2(image_msg, desired_encoding="bgr8")
            elif self.rgb_image_type is CompressedImage:
                rgb_image = self.bridge.compressed_imgmsg_to_cv2(image_msg)
            else:
                rospy.loginfo("No RGB image received")
                return
        except Exception as e:
            rospy.logwarn(f"Error converting RGB image: {e}")
            return

        # Convert depth image
        try:
            depth_image = self.bridge.imgmsg_to_cv2(depth_msg).astype(np.float32)
        except Exception as e:
            rospy.logwarn(f"Error converting depth image: {e}")
            return

        # Process depth image
        depth_image = np.nan_to_num(depth_image, nan=0.0)
        depth_image = np.clip(depth_image, 0.5, 13)

        # Perform prediction
        results = self.model_loader.predict(rgb_image)

        # Handle no predictions
        if not results[0].boxes:
            self.mask_image_publisher.publish_mask(rgb_image, None, frame_id)
            return

        # Process mask
        mask = results[0].masks.data[0].cpu().numpy().astype(np.uint8) * 255
        mask = keep_largest_component(mask)

        # Convert depth to point cloud
        point_cloud = depth_to_pointcloud_from_mask(depth_image, self.intrinsic_matrix, mask)

        # 2D edge detection
        points_2d = pointcloud_to_2d(point_cloud)
        point_cloud_2d, left_points, right_points = edge_detection_2d(points=points_2d)

        left_points = left_points[left_points[:, 1] <= 8]
        right_points = right_points[right_points[:, 1] <= 8]

        # Validate sufficient edge points
        if len(left_points) <= 5 or len(right_points) <= 5:
            return

        # Smooth left and right edge points
        def smooth_points(points, poly_order = 2):
            x, y = points[:, 0], points[:, 1]
            x_smooth = savgol_filter(x, len(x), poly_order)
            y_smooth = savgol_filter(y, len(y), poly_order)
            return np.column_stack((x_smooth, np.zeros_like(x_smooth), y_smooth))

        points_fine_l = smooth_points(left_points, 2)
        points_fine_r = smooth_points(right_points, 2)

        # Publish paths and masks
        self.left_path_publisher.publish_path(points_fine_l, frame_id)
        self.right_path_publisher.publish_path(points_fine_r, frame_id)
        self.mask_image_publisher.publish_mask(rgb_image, results, frame_id)

        self.path_publisher.publish_path([points_fine_l, points_fine_r], frame_id)

        #points_3d_left = np.hstack((left_points[:, [0]], np.zeros((left_points.shape[0], 1)), left_points[:, [1]]))
        #points_3d_right = np.hstack((right_points[:, [0]], np.zeros((right_points.shape[0], 1)), right_points[:, [1]]))

        #self.point_cloud_publisher.publish_pointcloud(point_cloud.points, points_3d_right, points_3d_left, frame_id)

        end_time = time.time()

        self.timers = [int((end_time - start_time) * 10 ** 9)] + self.timers [:-1]
        print(f"Function executed in {end_time - start_time:.6f} seconds")


    def run(self):
        """
        Spins the node to keep it running and listening to messages.
        """
        rospy.spin()
