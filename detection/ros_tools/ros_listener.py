import sys
sys.path.append("..")

import rospy
from sensor_msgs.msg import Image, CompressedImage, CameraInfo, Imu
from message_filters import Subscriber, TimeSynchronizer

from cv_bridge import CvBridge

import numpy as np

from detection.pointcloud.depth_to_pointcloud import depth_to_pointcloud_from_mask
from detection.pointcloud.pointcloud_edge_detection import edge_detection, edge_detection_2d, remove_edge_points
from detection.pointcloud.pointcloud_converter import pointcloud_to_2d

import detection.ros_tools.publisher.ros_path_publisher as ros_path_publisher
import detection.ros_tools.publisher.ros_mask_publisher as ros_mask_publisher
import detection.ros_tools.publisher.ros_pointcloud_publisher as ros_pointcloud_publisher
import detection.ros_tools.publisher.ros_road_lines_publisher as ros_road_lines_publisher

from detection.tools.mask import  keep_largest_component, get_mask_edge_distance

from scipy.spatial.transform import Rotation as R

import time

from scipy.signal import savgol_filter

import matplotlib.pyplot as plt

from skimage.restoration import inpaint_biharmonic



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


                print(self.intrinsic_matrix)
                print(msg)
        except rospy.ROSException as e:
            rospy.logerr(f"Failed to receive a message on {topic_name}: {e}")

    def quaternion_to_euler(self, q):
        """Convert quaternion to Euler angles (roll, pitch, yaw)."""
        rotation = R.from_quat([q.x, q.y, q.z, q.w])
        return rotation.as_euler('xyz', degrees=False)

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

        # Process depth image: replace NaNs with zeros and clip to valid range
        depth_image = np.nan_to_num(depth_image, nan=0.0)
        depth_image = np.clip(depth_image, 0.5, 13)

        # If depth and RGB dimensions differ, attempt a reshape (this is a no-op if dimensions already match)
        if rgb_image.shape[0] != depth_image.shape[0] or rgb_image.shape[1] != depth_image.shape[1]:
            depth_image = depth_image.reshape((rgb_image.shape[0], rgb_image.shape[1], 1))

        # Perform prediction using the loaded model
        results = self.model_loader.predict(rgb_image)

        # If no predictions are found, publish the mask and exit early
        if not results[0].boxes:
            self.mask_image_publisher.publish_mask(rgb_image, None, frame_id)
            return

        left_paths = []
        right_paths = []
        boxes = results[0].boxes.xyxy

        # Process each mask in the prediction results
        for i, mask_data in enumerate(results[0].masks.data):
            # Process mask: convert to uint8 and keep the largest connected component
            mask = (mask_data.cpu().numpy().astype(np.uint8) * 255)
            mask = keep_largest_component(mask)

            # Convert depth to point cloud from the mask
            point_cloud = depth_to_pointcloud_from_mask(depth_image, self.intrinsic_matrix, mask)

            # 2D edge detection
            points_2d = pointcloud_to_2d(point_cloud)
            left_points, right_points = edge_detection_2d(
                points_3D=np.asarray(point_cloud.points),
                points_2D=points_2d
            )

            # Filter edge points using the provided function
            filtered_left_points, filtered_right_points = remove_edge_points(
                mask, boxes[i], depth_image, left_points, right_points, self.intrinsic_matrix
            )

            # Validate that sufficient edge points exist
            if len(filtered_left_points) <= 2 or len(filtered_right_points) <= 2:
                return

            # Smooth left and right edge points using Savitzky–Golay filter
            def smooth_points(points, poly_order=2):
                x, y = points[:, 0], points[:, 1]
                x_smooth = savgol_filter(x, len(x), poly_order)
                y_smooth = savgol_filter(y, len(y), poly_order)
                # Return 3D points with y set to zero (or replaced by any other value if needed)
                return np.column_stack((x_smooth, np.zeros_like(x_smooth), y_smooth))

            points_fine_l = smooth_points(filtered_left_points[:, [0, 2]], poly_order=2)
            points_fine_r = smooth_points(filtered_right_points[:, [0, 2]], poly_order=2)

            left_paths.append(points_fine_l)
            right_paths.append(points_fine_r)

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

        self.left_road_lines_publisher.publish_path(left_paths, right_paths, frame_id)
        self.mask_image_publisher.publish_mask(rgb_image, results, frame_id)

        if False:
            #points_3d_left = np.hstack((filtered_left_points[:, [0]], np.zeros((filtered_left_points.shape[0], 1)), filtered_left_points[:, [1]]))
            #points_3d_right = np.hstack((filtered_right_points[:, [0]], np.zeros((filtered_right_points.shape[0], 1)), filtered_right_points[:, [1]]))

            self.point_cloud_publisher.publish_pointcloud(point_cloud.points, filtered_right_points, filtered_left_points, frame_id)

        end_time = time.time()

        self.timers = [int((end_time - start_time) * 10 ** 9)] + self.timers [:-1]
        print(f"Function executed in {end_time - start_time:.6f} seconds")


    def run(self):
        """
        Spins the node to keep it running and listening to messages.
        """
        rospy.spin()
