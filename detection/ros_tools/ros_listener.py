import sys
sys.path.append("..")

import rospy
from sensor_msgs.msg import Image, CompressedImage, CameraInfo, Imu
from message_filters import Subscriber, TimeSynchronizer

from cv_bridge import CvBridge

import cv2

import numpy as np

from detection.pointcloud.depth_to_pointcloud import depth_to_pointcloud_from_mask

import detection.ros_tools.publisher.ros_path_publisher as ros_path_publisher
import detection.ros_tools.publisher.ros_mask_publisher as ros_mask_publisher
import detection.ros_tools.publisher.ros_pointcloud_publisher as ros_pointcloud_publisher
import detection.ros_tools.publisher.ros_road_lines_publisher as ros_road_lines_publisher


import detection.utils.path_utils as path_utils
import detection.utils.pointcloud_utils as pointcloud_utils
import detection.utils.road_utils as road_utils
import detection.utils.boundary_identification_utils as boundary_identification_utils

import detection.utils.transformation_utils as transformation_utils

import detection.utils.timer_utils as timer_utils

import scipy.ndimage as nd



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

        self.pointcloud_previous= None

        self.max_depth = 13

        # Set timer for pause
        # TODO: Change for not static
        self.timers = [200000000, 200000000, 200000000, 200000000, 200000000]

        self.road_width = 2

        # Subscribers for the topics
        self.depth_sub = Subscriber('/hazard_front/zed_node_front/depth/depth_registered', Image)
        self.imu_sub = Subscriber('/hazard_front/zed_node_front/imu/data', Imu)


        self.get_available_image_topic()

        # Synchronize the topics using TimeSynchronizer
        self.ts = TimeSynchronizer([self.image_sub, self.depth_sub], 1)
        self.ts.registerCallback(self.callback)

        self.left_path_publisher = ros_path_publisher.SinglePathPublisher(topic_name='/path/left_path')
        self.right_path_publisher = ros_path_publisher.SinglePathPublisher(topic_name='/path/right_path')

        self.left_path_publisher_2 = ros_path_publisher.SinglePathPublisher(topic_name='/path/left_path_2')
        self.right_path_publisher_2 = ros_path_publisher.SinglePathPublisher(topic_name='/path/right_path_2')

        self.left_road_lines_publisher = ros_road_lines_publisher.RoadLinesPublisher(topic_name='/road/road_lines_topic')
        #self.right_road_lines_publisher = ros_road_lines_publisher.RoadLinesPublisher(topic_name='/road/right_road_lines_topic')

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
        except rospy.ROSException as e:
            rospy.logerr(f"Failed to receive a message on {topic_name}: {e}")

    def callback(self, image_msg, depth_msg):
        """
        Callback function that handles synchronized RGB and depth messages.

        This function:
          - Skips processing if a previous frame is still being processed.
          - Ensures the intrinsic matrix and image type are initialized.
          - Converts ROS image messages to OpenCV images.
          - Processes the depth image and performs prediction.
          - Extracts edge points from the predicted masks.
          - Smooths the left and right edge paths and publishes them.
        """
        global last_processed_time

        # Skip if still processing a previous frame
        wait_duration = rospy.Duration(0, int(sum(self.timers) / 5) + 50000000)
        if rospy.Time.now() - last_processed_time < wait_duration:
            return

        last_processed_time = rospy.Time.now()
        start_time = timer_utils.start_timer()

        # Initialize intrinsic matrix and image type if not already done
        if self.intrinsic_matrix is None:
            self.single_listen(self.camera_info_topic, CameraInfo)
            return

        if self.rgb_image_type is None:
            self.get_available_image_topic()
            return

        # Validate timestamps for synchronization
        if image_msg.header.stamp != depth_msg.header.stamp:
            rospy.loginfo("NO synchronized messages received")
            return

        frame_id = image_msg.header.frame_id
        time_stamp = image_msg.header.stamp

        # Convert RGB image
        try:
            # Check if we get compressed or uncompress image from ROS
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
            depth_image[depth_image > self.max_depth] = 0
            depth_image[np.isnan(depth_image)] = 0
        except Exception as e:
            rospy.logwarn(f"Error converting depth image: {e}")
            return

        # If depth and RGB dimensions differ, attempt a reshape (this is a no-op if dimensions already match)
        if rgb_image.shape[0] != depth_image.shape[0] or rgb_image.shape[1] != depth_image.shape[1]:
            depth_image = depth_image.reshape((rgb_image.shape[0], rgb_image.shape[1], 1))

        # Perform prediction using the loaded model
        masks_predicted, classes_predicted = self.model_loader.predict(rgb_image)

        # If no predictions are found, publish the mask and exit early
        if len(masks_predicted) == 0:
            rospy.logwarn(f"No masks, classes received")
            self.mask_image_publisher.publish_yolo_mask(rgb_image, None, None, None, frame_id)
            return

        paths = []
        masks = []
        road_masks = []

        kernel = np.ones((7, 7), np.uint8)
        #kernel = 3
        degree = 3  # Moved out of loop

        for i, mask in enumerate(masks_predicted):
            if classes_predicted[i] == 7:

                # Reduce the width of the mask with a kernel, that the pointcloud is not so width
                mask = cv2.erode(mask, kernel, iterations=1)

                # TODO: Test if this methode of fitting a line over the mask is better for the results
                #mask_coeffs, xmin, xmax  = mask_tool.fit_polynomial_to_mask(mask_out, degree=3, max_radius=600)
                #mask = mask_tool.rasterize_polynomial(mask.shape, mask_coeffs, xmin, xmax, 3)


                # Set Mask distance to the same as in the depth: self.max_dist
                mask[depth_image == 0] = 0
                masks.append(mask)

                # Get point cloud and filter NaNs
                point_cloud = depth_to_pointcloud_from_mask(depth_image, self.intrinsic_matrix, mask)
                points = np.asarray(point_cloud.points)

                # Pointcloud needs to have a minimum number of points: min 1 more than the degree for the line fitting
                if points.size == 0 or len(points) <= degree:
                    continue  # skip empty point clouds

                points = pointcloud_utils.filter_pointcloud_by_distance(points, distance=self.max_depth)
                points = points[~np.isnan(points).any(axis=1)]

                max_distance = pointcloud_utils.get_max_distance_from_pointcloud(points)

                if max_distance < 1:
                    rospy.loginfo(f"Remove Mask")
                    continue
                elif max_distance < 5:
                    rospy.loginfo(f"Max Distance < 5m: {max_distance}")
                    degree = 2
                else:
                    degree = 2

                x_fit, y_fit, z_fit = path_utils.calculate_fitted_line(points, degree, t_fit=50)

                # TODO: Ask Hamid if y should be zero or not
                #y_fit = np.zeros_like(x_fit)

                points_line = path_utils.ensure_first_point_closest_to_origin(np.column_stack((x_fit, y_fit, z_fit)))

                paths.append(points_line)
            else:
                road_masks.append(mask)

        path_pairs, path_np_extanded = boundary_identification_utils.find_left_to_right_pairs(np.asarray(paths), masks, road_masks, road_width=self.road_width)

        path_pairs = np.atleast_2d(path_pairs)
        if path_pairs.size > 0 and path_pairs.shape[1] > 0:
            left_paths = [path_np_extanded[i] if isinstance(i, int) else [] for i in path_pairs[:, 0]]
            right_paths = [path_np_extanded[i] if isinstance(i, int) else [] for i in path_pairs[:, 1]]
        else:
            left_paths = [[]]
            right_paths = [[]]

        new_road_width = road_utils.calculate_road_width(self.road_width, left_paths, right_paths)
        if new_road_width is None:
            return
        self.road_width = new_road_width


        ### Publish to ROS
        # Publish the first set of left/right paths
        self.left_path_publisher.publish_path(left_paths[0], frame_id)
        self.right_path_publisher.publish_path(right_paths[0], frame_id)

        # If more than one path exists, publish the second set; otherwise publish empty paths
        if len(left_paths) > 1:
            self.left_path_publisher_2.publish_path(left_paths[1], frame_id)
            self.right_path_publisher_2.publish_path(right_paths[1], frame_id)
        else:
            self.left_path_publisher_2.publish_path([], frame_id)
            self.right_path_publisher_2.publish_path([], frame_id)

        self.left_road_lines_publisher.publish_path(left_paths, right_paths, frame_id, time_stamp)
        self.mask_image_publisher.publish_yolo_mask(rgb_image, masks, [], path_pairs, frame_id, yolo_mask=False)


        # For testing: publish pointcloud from the mask -> slow down the detection when True
        if False:
            from detection.pointcloud.create_pointcloud import create_pointcloud

            pointcloud = create_pointcloud(depth_image, self.intrinsic_matrix)
            self.point_cloud_publisher.publish_pointcloud(pointcloud.points, [], [], frame_id)

        self.timers = timer_utils.end_timer(self.timers, start_time)


    def run(self, distance):
        """
        Spins the node to keep it running and listening to messages.
        """
        self.max_depth = float(distance)
        rospy.spin()
