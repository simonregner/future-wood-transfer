import sys
sys.path.append("..")

import rospy
from sensor_msgs.msg import Image, CompressedImage, CameraInfo
from message_filters import Subscriber, TimeSynchronizer

from cv_bridge import CvBridge

import numpy as np

from detection.pointcloud.depth_to_pointcloud import depth_to_pointcloud_from_mask
from detection.pointcloud.pointcloud_edge_detection import edge_detection

from detection.path.point_function import fit_line_3d_smooth

from detection.ros_tools.ros_publisher import PathPublisher, MaskPublisher, PointcloudPublisher

from detection.tools.mask import  keep_largest_component

import time

import cv2

last_processed_time = rospy.Time()

class TimeSyncListener():
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

        # Subscribers for the topics
        self.depth_sub = Subscriber('/hazard_front/zed_node_front/depth/depth_registered', Image)

        self.get_available_image_topic()

        # Synchronize the topics using TimeSynchronizer
        self.ts = TimeSynchronizer([self.image_sub, self.depth_sub], 3)
        self.ts.registerCallback(self.callback)

        self.left_path_publisher = PathPublisher(topic_name='/path/left_path')
        self.right_path_publisher = PathPublisher(topic_name='/path/right_path')

        self.mask_image_publisher = MaskPublisher(topic_name='/ml/mask_image')

        self.point_cloud_publisher = PointcloudPublisher(topic_name='/ml/pointcloud')

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
        if rospy.Time.now() - last_processed_time < rospy.Duration(0.1):
            return  # Skip this frame

        last_processed_time = rospy.Time.now()

        start_time = time.time()
        if self.intrinsic_matrix is None:
            self.single_listen(self.camera_info_topic, CameraInfo)

        # TODO: check if this is working
        if self.rgb_image_type is None:
            self.get_available_image_topic()
            self.ts.registerCallback(self.callback)
            return

        # Check if the timestamp is the same
        if image_msg.header.stamp.secs != depth_msg.header.stamp.secs:
            rospy.loginfo("NO synchronized messages received")
            rospy.loginfo(f"Image timestamp: {image_msg.header.stamp}")
            rospy.loginfo(f"Depth timestamp: {depth_msg.header.stamp}")
            return

        # TODO: Check
        frame_id = image_msg.header.frame_id

        # Convert correct image
        if self.rgb_image_type is Image:
            rgb_image = self.bridge.imgmsg_to_cv2(image_msg)
            rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGRA2BGR)
        elif self.rgb_image_type is CompressedImage:
            rgb_image = self.bridge.compressed_imgmsg_to_cv2(image_msg)
        else:
            rospy.loginfo(f"No RGB image received")
            return

        depth_image = self.bridge.imgmsg_to_cv2(depth_msg)
        depth_image = np.array(depth_image, dtype=np.float32)

        # Replace NaN values with 0 (or another appropriate value)
        depth_array = np.nan_to_num(depth_image, nan=0.0)

        # Clip negative values to 0
        depth_array[depth_array < 1.5] = 0
        depth_array[depth_array > 10] = 0

        results = self.model_loader.predict(rgb_image)

        # check if there is a prediction in the image
        if len(results[0].boxes) == 0:
            self.mask_image_publisher.publish_mask(rgb_image, None, frame_id)
            return

        mask = results[0].masks.data[0].cpu().numpy().astype(np.uint8) * 255

        mask = keep_largest_component(mask)

        point_cloud = depth_to_pointcloud_from_mask(
            depth_image=depth_array,
            intrinsic_matrix=self.intrinsic_matrix, mask=mask)

        #point_cloud, ind = point_cloud.remove_statistical_outlier(nb_neighbors=10, std_ratio=2.0)

        point_cloud, left_points, right_points = edge_detection(point_cloud=point_cloud)
        #point_cloud, left_points, right_points = split_pointcloud(point_cloud=point_cloud)


        if len(left_points) == 0 or len(right_points) == 0:
            return

        # Fit a line to the edge of the pointcloud
        x_fine_l, y_fine_l, z_fine_l = fit_line_3d_smooth(points=left_points, smoothing_factor=1)
        x_fine_r, y_fine_r, z_fine_r = fit_line_3d_smooth(points=right_points, smoothing_factor=1)

        # Combine the x,y,z point to a single stack
        points_fine_l = np.vstack((x_fine_l, y_fine_l, z_fine_l)).T
        points_fine_r = np.vstack((x_fine_r, y_fine_r, z_fine_r)).T

        # Publish information to ROS
        self.left_path_publisher.publish_path(points_fine_l[:-5], frame_id)
        self.right_path_publisher.publish_path(points_fine_r[:-5], frame_id)

        self.mask_image_publisher.publish_mask(rgb_image, results, frame_id)
        self.point_cloud_publisher.publish_pointcloud(np.asarray(point_cloud.points), right_points, left_points, frame_id)

        end_time = time.time()
        print(f"Function executed in {end_time - start_time:.6f} seconds")


    def run(self):
        """
        Spins the node to keep it running and listening to messages.
        """
        rospy.spin()
