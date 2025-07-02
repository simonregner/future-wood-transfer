import sys
sys.path.append("..")

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CompressedImage, CameraInfo, Imu
from message_filters import Subscriber, TimeSynchronizer, ApproximateTimeSynchronizer
from cv_bridge import CvBridge
import cv2
import numpy as np
from std_msgs.msg import Header

# --- Your own module imports ---
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

last_processed_time = 0  # use float seconds for timing in ROS2

class TimeSyncListener(Node):
    def __init__(self, model_loader, max_depth=13):
        super().__init__('time_sync_listener')

        self.model_loader = model_loader
        self.max_depth = max_depth

        self.intrinsic_matrix = None
        self.camera_info_topic = "/hazard_front/stereo_node_front/rgb/camera_info"
        self.bridge = CvBridge()
        self.rgb_image_type = None
        self.pointcloud_previous = None
        self.timer = [200000000, 200000000, 200000000, 200000000, 200000000]  # in nanoseconds
        self.road_width = 2

        # Subscribers
        self.image_sub = Subscriber(self, Image, '/hazard_front/stereo_node_front/rgb')
        self.depth_sub = Subscriber(self, Image, '/hazard_front/stereo_node_front/depth')
        self.imu_sub = Subscriber(self, Imu, '/hazard_front/stereo_node_front/imu/data')



        # ---- Message Filter Synchronization ----
        self.ts = TimeSynchronizer(
            [self.image_sub, self.depth_sub],
            queue_size=10,
        )
        self.ts.registerCallback(self.callback)

        # Publishers
        self.left_path_publisher = ros_path_publisher.SinglePathPublisher(topic_name='/path/left_path')
        self.right_path_publisher = ros_path_publisher.SinglePathPublisher(topic_name='/path/right_path')
        self.left_path_publisher_2 = ros_path_publisher.SinglePathPublisher(topic_name='/path/left_path_2')
        self.right_path_publisher_2 = ros_path_publisher.SinglePathPublisher(topic_name='/path/right_path_2')
        self.road_lines_publisher = ros_road_lines_publisher.RoadLinesPublisher(topic_name='/road/road_lines_topic')
        self.mask_image_publisher = ros_mask_publisher.MaskPublisher(topic_name='/ml/mask_image')
        self.point_cloud_publisher = ros_pointcloud_publisher.PointcloudPublisher(topic_name='/ml/pointcloud')

        self.get_logger().info(
            f"TimeSyncListener initialized and running with max_depth {self.max_depth}m"
        )

        # Async CameraInfo initialization
        self._camera_info_received = False
        self._camera_info_subscription = self.create_subscription(
            CameraInfo,
            self.camera_info_topic,
            self._camera_info_callback,
            10
        )

    def get_available_image_topic(self):
        # Use rclpy API to discover topics
        # (There is no direct equivalent of rospy.get_published_topics in rclpy, but you can use get_topic_names_and_types)
        available_topics = [topic for topic, _ in self.get_topic_names_and_types()]
        compressed_topic = "/hazard_front/stereo_node_front/rgb/compressed"
        uncompressed_topic = "/hazard_front/stereo_node_front/rgb"

        if compressed_topic in available_topics:
            self.get_logger().info(f"Topic found: {compressed_topic}")
            self.image_sub = Subscriber(self, CompressedImage, compressed_topic)
            self.rgb_image_type = CompressedImage
            return
        elif uncompressed_topic in available_topics:
            self.get_logger().info(f"Topic found: {uncompressed_topic}")
            self.image_sub = Subscriber(self, Image, uncompressed_topic)
            self.rgb_image_type = Image
            return
        else:
            self.get_logger().error("Neither compressed nor uncompressed image topic is available!")
            rclpy.shutdown()
            return

    def _camera_info_callback(self, msg):
        self.intrinsic_matrix = np.array(msg.k).reshape(3, 3)
        self._camera_info_received = True
        self.get_logger().info(f"CameraInfo received. Intrinsic matrix: {self.intrinsic_matrix}")
        # Once received, unsubscribe
        self.destroy_subscription(self._camera_info_subscription)
        self._camera_info_subscription = None

    def callback(self, image_msg, depth_msg):
        global last_processed_time
        now = self.get_clock().now().nanoseconds

        # Replace ROS1 rospy.Duration with nanoseconds math
        wait_duration = int(sum(self.timer) / 5) + 50000000  # nanoseconds
        if (now - last_processed_time) < wait_duration:
            return
        last_processed_time = now
        #start_time = timer_utils.start_timer()

        # Initialize intrinsic matrix and image type if not already done
        if self.intrinsic_matrix is None:
            if not self._camera_info_received:
                self.get_logger().warn("Waiting for CameraInfo message...")
            return

        if self.rgb_image_type is None:
            self.get_available_image_topic()
            return

        # Validate timestamps for synchronization
        if image_msg.header.stamp != depth_msg.header.stamp:
            self.get_logger().info("NO synchronized messages received")
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
                self.get_logger().info("No RGB image received")
                return
        except Exception as e:
            self.get_logger().warn(f"Error converting RGB image: {e}")
            return

        # Convert depth image
        try:
            depth_image = self.bridge.imgmsg_to_cv2(depth_msg).astype(np.float32)
            if depth_msg.encoding == "16UC1":
                depth_image = depth_image / 1000
            depth_image[depth_image > self.max_depth] = 0
            depth_image[np.isnan(depth_image)] = 0
        except Exception as e:
            self.get_logger().warn(f"Error converting depth image: {e}")
            return

        if rgb_image.shape[0] != depth_image.shape[0] or rgb_image.shape[1] != depth_image.shape[1]:
            depth_image = depth_image.reshape((rgb_image.shape[0], rgb_image.shape[1], 1))

        # Perform prediction using the loaded model
        masks_predicted, classes_predicted = self.model_loader.predict(rgb_image)
        if len(masks_predicted) == 0:
            self.get_logger().warn("No masks, classes received")
            self.mask_image_publisher.publish_yolo_mask(rgb_image, None, None, None, frame_id)
            return

        paths = []
        masks = []
        road_masks = []
        kernel = np.ones((3, 3), np.uint8)
        degree = 3

        for i, mask in enumerate(masks_predicted):
            if classes_predicted[i] == 7:
                mask = cv2.erode(mask, kernel, iterations=1)
                mask[depth_image == 0] = 0
                masks.append(mask)
                point_cloud = depth_to_pointcloud_from_mask(depth_image, self.intrinsic_matrix, mask)
                points = np.asarray(point_cloud.points)
                if points.size == 0 or len(points) <= degree:
                    continue
                points = pointcloud_utils.filter_pointcloud_by_distance(points, distance=self.max_depth)
                points = points[~np.isnan(points).any(axis=1)]
                max_distance = pointcloud_utils.get_max_distance_from_pointcloud(points)
                if max_distance < 1:
                    self.get_logger().info("Remove Mask")
                    continue
                elif max_distance < 5:
                    self.get_logger().info(f"Max Distance < 5m: {max_distance}")
                    degree = 2
                else:
                    degree = 2
                x_fit, y_fit, z_fit = path_utils.calculate_fitted_line(points, degree, t_fit=50)
                points_line = path_utils.ensure_first_point_closest_to_origin(np.column_stack((x_fit, y_fit, z_fit)))
                paths.append(points_line)
            else:
                continue
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

        # --- Publish to ROS2 ---
        self.left_path_publisher.publish_path(left_paths[0], frame_id)
        self.right_path_publisher.publish_path(right_paths[0], frame_id)
        if len(left_paths) > 1:
            self.left_path_publisher_2.publish_path(left_paths[1], frame_id)
            self.right_path_publisher_2.publish_path(right_paths[1], frame_id)
        else:
            self.left_path_publisher_2.publish_path([], frame_id)
            self.right_path_publisher_2.publish_path([], frame_id)

        self.road_lines_publisher.publish_path(left_paths, right_paths, frame_id, time_stamp)
        self.mask_image_publisher.publish_yolo_mask(rgb_image, masks, road_masks, path_pairs, frame_id, yolo_mask=False)

        # Optionally publish pointcloud
        if False:
            from detection.pointcloud.create_pointcloud import create_pointcloud
            pointcloud = create_pointcloud(depth_image, self.intrinsic_matrix)
            self.point_cloud_publisher.publish_pointcloud(pointcloud.points, [], [], frame_id)

        #self.timers = timer_utils.end_timer(self.timers, start_time)

    def run(self, distance):
        self.max_depth = float(distance)
        rclpy.spin(self)

# USAGE:
# rclpy.init()
# model_loader = ... # your ML model class instance
# node = TimeSyncListener(model_loader)
# node.run(distance=13)
# node.destroy_node()
# rclpy.shutdown()
