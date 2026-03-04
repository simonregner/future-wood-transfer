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
import detection.ros.publishers.path_publisher as ros_path_publisher
import detection.ros.publishers.mask_publisher as ros_mask_publisher
import detection.ros.publishers.pointcloud_publisher as ros_pointcloud_publisher
import detection.ros.publishers.road_lines_publisher as ros_road_lines_publisher
import detection.path.path_utils as path_utils
import detection.pointcloud.pointcloud_utils as pointcloud_utils
import detection.road.road_utils as road_utils
import detection.path.boundary_identification as boundary_identification_utils
import detection.utils.transformation_utils as transformation_utils
import detection.utils.timer_utils as timer_utils
import scipy.ndimage as nd

# Tracks the ROS2 timestamp (nanoseconds) of the last processed frame.
# Used to throttle callbacks so we don't process every incoming frame pair.
last_processed_time = 0

class TimeSyncListener(Node):
    """
    A ROS2 node that synchronizes RGB and depth images, performs model inference to detect paths and road features,
    and publishes results such as paths, masks, and pointclouds.

    Attributes:
        model_loader: Object that provides the `.predict()` method to get segmentation masks and classes.
        max_depth (float): Maximum depth distance in meters to consider.
        computation_type (int): Type of computation (0 = all, 1 = only road lines).
    """
    def __init__(self, model_loader, args):
        """
        Initializes the TimeSyncListener node, sets up subscribers, publishers, and loads the camera intrinsics.

        Args:
            model_loader: A model object capable of performing inference on RGB images.
            max_depth (float): Maximum depth value to consider in meters (default is 13).
            computation_type (int): Processing mode; 0 = all, 1 = road lines only.
        """
        super().__init__('time_sync_listener')

        self.get_logger().info(
            f"Start initialization of TimeSyncListener ..."
        )

        self.args = args

        self.model_loader = model_loader
        self.max_depth = self.args.max_depth

        self.intrinsic_matrix = None
        self.camera_info_topic = self.args.topic_camera_info
        self.bridge = CvBridge()

        self.rgb_image_type = Image
        self.pointcloud_previous = None
        # Minimum time between processed frames in nanoseconds (200ms = 5 Hz max).
        # Updated dynamically by timer_utils to match actual processing time.
        self.timers_subscriber = 200000000
        self.road_width = 2  # meters; may be updated dynamically based on detected road

        # Subscribers
        self.get_logger().info(
            f"Initialization of Subscribers and Time Synchronization"
        )
        if self.args.rgb_image_type == "Image":
            self.get_logger().info(
                f"Using raw RGB image topic: {self.args.topic_rgb}"
            )
            self.rgb_image_type = Image#
        else:
            self.get_logger().info(
                f"Using compressed RGB image topic: {self.args.topic_rgb}"
            )
            self.rgb_image_type = CompressedImage


        self.image_sub = Subscriber(self, self.rgb_image_type, self.args.topic_rgb)
        self.depth_sub = Subscriber(self, Image, self.args.topic_depth)

        # ---- Message Filter Synchronization ----
        if self.args.time_syncronizer:
            
            self.ts = TimeSynchronizer(
                [self.image_sub, self.depth_sub],
                queue_size=self.args.time_sync_queue_size,
            )
            self.ts.registerCallback(self.on_synced_rgb_depth)
            self.get_logger().info(
                f"Init TimeSynchronizer with queue size: {self.args.time_sync_queue_size}"
            )
        else:
            self.ts = ApproximateTimeSynchronizer(
                [self.image_sub, self.depth_sub],
                self.args.time_sync_queue_size,
                self.args.sync_max_delay,  # seconds
            )
            self.ts.registerCallback(self.on_synced_rgb_depth)
            self.get_logger().info(
                f"Init ApproximateTimeSynchronizer with queue size: {self.args.time_sync_queue_size} and max delay: {self.args.sync_max_delay} seconds"
            )

        # Initialize Publisher
        self.get_logger().info(
            f"Initialization of Publishers"
        )
        self.left_path_publisher = ros_path_publisher.SinglePathPublisher(topic_name='/path/left_path')
        self.right_path_publisher = ros_path_publisher.SinglePathPublisher(topic_name='/path/right_path')
        self.left_path_publisher_2 = ros_path_publisher.SinglePathPublisher(topic_name='/path/left_path_2')
        self.right_path_publisher_2 = ros_path_publisher.SinglePathPublisher(topic_name='/path/right_path_2')
        self.road_lines_publisher = ros_road_lines_publisher.RoadLinesPublisher(topic_name='/road/road_lines_topic')
        self.mask_image_publisher = ros_mask_publisher.MaskPublisher(topic_name='/ml/mask_image')
        self.point_cloud_publisher = ros_pointcloud_publisher.PointcloudPublisher(topic_name='/ml/pointcloud')

        # Async CameraInfo initialization
        self._camera_info_received = False
        self._camera_info_subscription = self.create_subscription(
            CameraInfo,
            self.camera_info_topic,
            self._camera_info_callback,
            10
        )

        self.get_logger().info(
            f"TimeSyncListener initialized and running with max_depth {self.max_depth}m"
        )
        self.get_logger().info(
            f"Waiting for Camera Intrinsic matrix from topic: {self.camera_info_topic}"
        )


    def on_synced_rgb_depth(self, image_msg, depth_msg):
        """
        Callback fired when a synchronized RGB + depth image pair arrives.
        Performs image conversion, model inference, mask filtering, 3D path fitting,
        and publishes results (paths, masks, pointclouds) to their ROS topics.

        Args:
            image_msg (sensor_msgs.msg.Image or CompressedImage): The RGB image message.
            depth_msg (sensor_msgs.msg.Image): The depth image message.
        """
        global last_processed_time
        now = self.get_clock().now().nanoseconds

        # Throttle: skip this frame if the last processed frame is too recent.
        # wait_duration adds a 50ms buffer on top of the previous processing time
        # so the node self-regulates to stay within its own processing capacity.
        wait_duration = int(self.timers_subscriber) + 50000000  # nanoseconds
        if (now - last_processed_time) < wait_duration:
            return
        last_processed_time = now
        start_time = timer_utils.start_timer()

        # Block processing until the camera intrinsic matrix has been received
        # via the /camera_info topic (populated by _camera_info_callback).
        if self.intrinsic_matrix is None:
            if not self._camera_info_received:
                self.get_logger().warn(f"Waiting for CameraInfo message, retry at next callback...")
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
            # 16UC1 encoding stores depth in millimeters → convert to meters
            if depth_msg.encoding == "16UC1":
                depth_image = depth_image / 1000
            # Zero out pixels beyond max range and invalid (NaN) readings
            depth_image[depth_image > self.max_depth] = 0
            depth_image[np.isnan(depth_image)] = 0
        except Exception as e:
            self.get_logger().warn(f"Error converting depth image: {e}")
            return

        # If shapes differ (e.g. different camera resolutions), reshape depth to match RGB.
        # This can happen when depth is registered to a different camera frame.
        if rgb_image.shape[0] != depth_image.shape[0] or rgb_image.shape[1] != depth_image.shape[1]:
            depth_image = depth_image.reshape((rgb_image.shape[0], rgb_image.shape[1], 1))

        # Perform prediction using the loaded model
        masks_predicted, classes_predicted = self.model_loader.predict(rgb_image)
        if len(masks_predicted) == 0:
            self.get_logger().warn("No masks, classes received")
            self.mask_image_publisher.publish_segmentation_mask(rgb_image, None, None, None, frame_id)
            return

        paths = []        # fitted 3D polylines, one per detected path boundary
        masks = []        # 2D binary masks for class-7 (path boundary) detections
        road_masks = []   # 2D binary masks for road-surface detections (currently unused)
        kernel = np.ones((3, 3), np.uint8)  # morphology kernel for erosion
        degree = 3        # polynomial degree for curve fitting (reduced to 2 for short paths)

        line_pointclouds = []  # 3D point clouds projected from each boundary mask

        for i, mask in enumerate(masks_predicted):
            # Class 7 = path boundary (e.g. forest track edge).
            # All other classes are skipped (road surface handling is disabled below).
            if classes_predicted[i] == 7:
                # Erode the mask by 1 pixel to remove thin noise along edges.
                mask = cv2.erode(mask, kernel, iterations=1)
                # Zero out mask pixels where depth is unknown (depth == 0).
                mask[depth_image == 0] = 0
                masks.append(mask)

                # Project the masked depth pixels into 3D space using camera intrinsics.
                point_cloud = depth_to_pointcloud_from_mask(depth_image, self.intrinsic_matrix, mask)
                line_pointclouds.append(point_cloud)
                points = np.asarray(point_cloud.points)

                # Skip masks that yield too few 3D points to fit a polynomial.
                if points.size == 0 or len(points) <= degree:
                    continue

                points = pointcloud_utils.filter_pointcloud_by_distance(points, distance=self.max_depth)
                points = points[~np.isnan(points).any(axis=1)]  # remove any remaining NaN rows

                # Use max distance as a quality gate:
                # < 1m → likely noise, discard;  < 5m → use lower polynomial degree.
                max_distance = pointcloud_utils.get_max_distance_from_pointcloud(points)
                if max_distance < 1:
                    self.get_logger().info("Remove Mask")
                    continue
                elif max_distance < 5:
                    self.get_logger().info(f"Max Distance < 5m: {max_distance}")
                    degree = 2
                else:
                    degree = 2

                # Fit a smooth polynomial curve through the 3D boundary points
                # and sample it at 50 evenly spaced parameter values.
                x_fit, y_fit, z_fit = path_utils.fit_parametric_polynomial_3d(points, degree, t_fit=50)

                # Reorder the fitted line so the closest point to the camera comes first
                # (needed for consistent left/right pairing downstream).
                points_line = path_utils.ensure_first_point_closest_to_origin(np.stack([x_fit, y_fit, z_fit], axis=1))
                paths.append(points_line)
            else:
                # NOTE: road_masks.append is intentionally unreachable here
                # (the `continue` above it skips all non-class-7 detections).
                continue
                road_masks.append(mask)

        # --- Path Pairing ---
        # Match detected boundary lines into left/right pairs based on spatial proximity
        # and the expected road width. Returns index pairs and the interpolated path array.
        path_pairs, path_np_extanded = boundary_identification_utils.find_left_to_right_pairs(np.asarray(paths), masks, road_masks, road_width=self.road_width)
        path_pairs = np.atleast_2d(path_pairs)

        # Unpack the paired indices into separate left and right path lists.
        # If no pairs were found, fall back to empty lists to avoid downstream errors.
        if path_pairs.size > 0 and path_pairs.shape[1] > 0:
            left_paths = [path_np_extanded[i] if isinstance(i, int) else [] for i in path_pairs[:, 0]]
            right_paths = [path_np_extanded[i] if isinstance(i, int) else [] for i in path_pairs[:, 1]]
        else:
            left_paths = [[]]
            right_paths = [[]]

        # Optionally update road_width from the detected geometry each frame.
        # Returns None if width cannot be determined (e.g. too few points).
        new_road_width = road_utils.calculate_road_width(self.road_width, left_paths, right_paths)
        if new_road_width is None:
            return
        # self.road_width = new_road_width  # disabled: keeps width fixed for stability

        # Always publish road boundary lines (used for visualization / downstream nav).
        self.road_lines_publisher.publish_path(left_paths, right_paths, frame_id, time_stamp)

        # Conditionally publish outputs based on the computation_type config flag.
        # This allows disabling expensive outputs (e.g. pointcloud) at runtime.
        if 'mask' in self.args.computation_type:
            self.mask_image_publisher.publish_segmentation_mask(rgb_image, masks, road_masks, path_pairs, frame_id, yolo_mask=False)

        if 'path' in self.args.computation_type:
            # Publish only the first (primary) left/right pair; secondary publishers are cleared.
            self.left_path_publisher.publish_path(left_paths[0], frame_id)
            self.right_path_publisher.publish_path(right_paths[0], frame_id)
            self.left_path_publisher_2.publish_path([], frame_id)
            self.right_path_publisher_2.publish_path([], frame_id)

        if 'pointcloud' in self.args.computation_type:
            # Lazy import to avoid loading heavy dependencies when pointcloud is disabled.
            from detection.pointcloud.create_pointcloud import create_pointcloud
            pointcloud = create_pointcloud(depth_image, self.intrinsic_matrix)
            self.point_cloud_publisher.publish_pointcloud(pointcloud.points, line_pointclouds[0], line_pointclouds[1], frame_id)

        # Update the throttle timer with the actual elapsed time so the node
        # dynamically adapts its processing rate to its own compute budget.
        self.timers_subscriber = timer_utils.end_timer(self.timers_subscriber, start_time)

    def _camera_info_callback(self, msg):
        """
        Callback to handle incoming CameraInfo messages and extract the intrinsic matrix.

        Args:
            msg (sensor_msgs.msg.CameraInfo): The message containing camera calibration data.
        """
        self.intrinsic_matrix = np.array(msg.k).reshape(3, 3)
        self._camera_info_received = True
        self.get_logger().info(f"CameraInfo received. Intrinsic matrix: {self.intrinsic_matrix}")
        # Once received, unsubscribe
        self.destroy_subscription(self._camera_info_subscription)
        self._camera_info_subscription = None

    def run(self, distance):
        self.max_depth = float(distance)
        rclpy.spin(self)