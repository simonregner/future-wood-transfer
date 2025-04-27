import sys

from numpy.matlib import empty

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

from detection.tools.mask import  remove_smaller_parts
from detection.tools.line_indentification import find_left_to_right_pairs

from detection.tools.road_information import  get_road_width


from scipy.spatial.transform import Rotation as R

import time
import scipy.ndimage as nd

from sklearn.decomposition import PCA

from scipy.spatial import ConvexHull
from itertools import combinations

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

        self.max_depth = 20

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

                print("Camera Distortion: ", msg.D)


                print(self.intrinsic_matrix)
                print(msg)
        except rospy.ROSException as e:
            rospy.logerr(f"Failed to receive a message on {topic_name}: {e}")

    def quaternion_to_euler(self, q):
        """Convert quaternion to Euler angles (roll, pitch, yaw)."""
        rotation = R.from_quat([q.x, q.y, q.z, q.w])
        return rotation.as_euler('xyz', degrees=False)

    def ensure_first_point_closest_to_origin(self, points) -> np.array:
        points = list(points)  # in case it's a numpy array or tuple
        first = np.array(points[0])
        last = np.array(points[-1])
        origin = np.array([0, 0, 0])

        if np.linalg.norm(last - origin) < np.linalg.norm(first - origin):
            return points[::-1]  # reversed
        return points

    def pointscloud_distance(self, points, distance=10):
        distances = np.linalg.norm(points, axis=1)

        # Keep only points within distance 10
        filtered_points = points[distances <= distance]

        return filtered_points

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
        time_stamp = image_msg.header.stamp

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

        nan_mask = np.isnan(depth_image)
        _, indices = nd.distance_transform_edt(nan_mask, return_distances=True, return_indices=True)
        #depth_image = depth_image[tuple(indices)]

        # If depth and RGB dimensions differ, attempt a reshape (this is a no-op if dimensions already match)
        if rgb_image.shape[0] != depth_image.shape[0] or rgb_image.shape[1] != depth_image.shape[1]:
            depth_image = depth_image.reshape((rgb_image.shape[0], rgb_image.shape[1], 1))

        # Perform prediction using the loaded model
        results = self.model_loader.predict(rgb_image)

        # If no predictions are found, publish the mask and exit early
        if not results[0].boxes:
            self.mask_image_publisher.publish_yolo_mask(rgb_image, None, None, None, frame_id)
            return


        paths = []
        masks = []
        road_masks = []

        kernel = np.ones((7, 7), np.uint8)
        degree = 3  # Moved out of loop

        # Precompute t_fit
        t_fit_len = 50

        for i, mask_data in enumerate(results[0].masks.data):
            if results[0].boxes.cls[i].cpu().numpy().astype(np.uint8) == 7:
                # Convert mask to uint8 and extract largest component
                mask = (mask_data.cpu().numpy().astype(np.uint8) * 255)
                #mask = remove_smaller_parts(mask, min_size=2000)

                mask = cv2.erode(mask, kernel, iterations=1)
                masks.append(mask)

                # Get point cloud and filter NaNs
                point_cloud = depth_to_pointcloud_from_mask(depth_image, self.intrinsic_matrix, mask)
                points = np.asarray(point_cloud.points)
                if points.size == 0:
                    continue  # skip empty point clouds

                points = self.pointscloud_distance(points, distance=20)

                points = points[~np.isnan(points).any(axis=1)]
                if len(points) < degree + 1:
                    continue  # Not enough points to fit polynomial

                min_point = np.min(points, axis=0)
                max_point = np.max(points, axis=0)
                max_distance = np.linalg.norm(max_point - min_point)

                if max_distance < 1:
                    rospy.logwarn(f"Remove Mask")
                    continue

                if max_distance < 5:
                    rospy.logwarn(f"Max Distance < 5m: {max_distance}")
                    degree = 2
                else:
                    degree = 3

                # PCA to sort points
                t = PCA(n_components=1).fit_transform(points).flatten()
                sorted_idx = np.argsort(t)
                points_sorted = points[sorted_idx]
                t_sorted = t[sorted_idx]

                # Fit polynomial for each coordinate using t_sorted as the parameter:
                # (you might want to adjust degrees as needed)
                poly_x = np.polyfit(t_sorted, points_sorted[:, 0], degree)
                poly_y = np.polyfit(t_sorted, points_sorted[:, 1], 2)
                poly_z = np.polyfit(t_sorted, points_sorted[:, 2], degree)

                # Generate the main fitted t-values for your current range:
                t_fit_main = np.linspace(t_sorted[0], t_sorted[-1], t_fit_len)
                x_fit_main = np.polyval(poly_x, t_fit_main)
                y_fit_main = np.polyval(poly_y, t_fit_main)
                #y_fit_main = np.zeros(t_fit_len)
                z_fit_main = np.polyval(poly_z, t_fit_main)

                # Estimate a spacing for the t-values.
                # One simple approach is to use the difference between the first two sorted t-values.
                # (You may also compute the median of t differences if that's more robust.)
                dt = t_sorted[1] - t_sorted[0]

                n_extension = 0

                # Generate n_extension extra t-values that extend *before* the beginning of your data.
                # For example, if you have 5 extra points, you can create them from t_sorted[0] - 5*dt up to t_sorted[0]
                t_fit_ext = np.linspace(t_sorted[0] - n_extension * dt, t_sorted[0], n_extension, endpoint=False)

                # Evaluate the fitted polynomial on the extension t-values:
                x_fit_ext = np.polyval(poly_x, t_fit_ext)
                y_fit_ext = np.polyval(poly_y, t_fit_ext)
                z_fit_ext = np.polyval(poly_z, t_fit_ext)

                # Option 1: If you want to keep them separate, you now have:
                #   x_fit_ext, y_fit_ext, z_fit_ext  => extension points
                #   x_fit_main, y_fit_main, z_fit_main => original fitted points

                # Option 2: If you want to combine the extension and the fitted points:
                x_fit = np.concatenate([x_fit_ext, x_fit_main])
                y_fit = np.concatenate([y_fit_ext, y_fit_main])
                z_fit = np.concatenate([z_fit_ext, z_fit_main])

                points_line = self.ensure_first_point_closest_to_origin(np.column_stack((x_fit, y_fit, z_fit)))
                paths.append(points_line)
            else:
                continue
                mask = (mask_data.cpu().numpy().astype(np.uint8) * 255)
                road_masks.append(mask)

        paths_np = np.asarray(paths)

        path_pairs, path_np_extanded = find_left_to_right_pairs(paths_np, masks, road_width=self.road_width)

        path_pairs = np.atleast_2d(path_pairs)
        if path_pairs.size > 0 and path_pairs.shape[1] > 0:
            left_paths = [path_np_extanded[i] if isinstance(i, int) else [] for i in path_pairs[:, 0]]
            right_paths = [path_np_extanded[i] if isinstance(i, int) else [] for i in path_pairs[:, 1]]

        else:
            left_paths = [[]]
            right_paths = [[]]
        #left_paths = [paths_np[i] if isinstance(i, int) else [] for i in path_pairs[:, 0]]
        #right_paths = [paths_np[i] if isinstance(i, int) else [] for i in path_pairs[:, 1]]

        for i in range(path_pairs.shape[0]):
            if len(left_paths[i]) == 0 or len(right_paths[i]) == 0 :
                rospy.logerr(f"Path Pair empty")
                continue
            #print(left_paths[i], right_paths[i])
            self.road_width = (self.road_width + get_road_width(left_paths[i], right_paths[i])) / 2

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
        self.mask_image_publisher.publish_yolo_mask(rgb_image, masks, road_masks, path_pairs, frame_id, yolo_mask=False)


        if False:
            from detection.pointcloud.create_pointcloud import create_pointcloud

            pointcloud = create_pointcloud(depth_image, self.intrinsic_matrix)
            self.point_cloud_publisher.publish_pointcloud(pointcloud.points, [], [], frame_id)

        end_time = time.time()

        self.timers = [int((end_time - start_time) * 10 ** 9)] + self.timers [:-1]
        print(f"Function executed in {end_time - start_time:.6f} seconds")


    def run(self):
        """
        Spins the node to keep it running and listening to messages.
        """
        rospy.spin()
