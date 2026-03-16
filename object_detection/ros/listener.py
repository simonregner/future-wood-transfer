import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CompressedImage, CameraInfo
from message_filters import Subscriber, TimeSynchronizer, ApproximateTimeSynchronizer
from cv_bridge import CvBridge
import cv2
import numpy as np

from ros.publishers.detection_publisher import DetectionPublisher

# Timestamp (ns) of the last frame we actually processed, used for throttling.
last_processed_time = 0


class ObjectDetectionListener(Node):
    """
    ROS2 node that time-synchronises RGB and depth images, runs a YOLO
    detection model on each pair, computes the 3-D position of every detected
    object and publishes results.

    Published topics:
        /object_detection/detections  (object_detection_msg/DetectedObjectsMsg)
        /object_detection/image       (sensor_msgs/Image) – annotated RGB
    """

    def __init__(self, model_loader, args):
        super().__init__('object_detection_listener')

        self.args = args
        self.model_loader = model_loader
        self.max_depth = args.max_depth
        self.bridge = CvBridge()
        self.intrinsic_matrix = None
        self._camera_info_received = False

        # Minimum gap between processed frames (ns).  Updated dynamically.
        self.min_frame_gap_ns = 200_000_000  # 200 ms → 5 Hz max

        # Filter detections to these COCO class IDs (None = all classes).
        # Default: person (0), car (2), truck (7), bus (5), motorcycle (3)
        self.target_classes = getattr(args, 'target_classes', [0, 2, 3, 5, 7])
        self.publish_markers = getattr(args, 'publish_markers', True)
        self.publish_mask = getattr(args, 'publish_mask', True)

        self.get_logger().info("Initialising ObjectDetectionListener …")

        # --- Subscribers ---
        rgb_type_str = getattr(args, 'rgb_image_type', 'Image')
        self.rgb_msg_type = Image if rgb_type_str == 'Image' else CompressedImage
        self.get_logger().info(
            f"RGB topic: {args.topic_rgb} ({rgb_type_str})"
        )

        self.image_sub = Subscriber(self, self.rgb_msg_type, args.topic_rgb)
        self.depth_sub = Subscriber(self, Image, args.topic_depth)

        if args.time_syncronizer:
            self.ts = TimeSynchronizer(
                [self.image_sub, self.depth_sub],
                queue_size=args.time_sync_queue_size,
            )
            self.get_logger().info(
                f"TimeSynchronizer queue={args.time_sync_queue_size}"
            )
        else:
            self.ts = ApproximateTimeSynchronizer(
                [self.image_sub, self.depth_sub],
                args.time_sync_queue_size,
                args.sync_max_delay,
            )
            self.get_logger().info(
                f"ApproximateTimeSynchronizer queue={args.time_sync_queue_size} "
                f"delay≤{args.sync_max_delay}s"
            )

        self.ts.registerCallback(self.on_synced_rgb_depth)

        # --- Camera info ---
        self._camera_info_sub = self.create_subscription(
            CameraInfo, args.topic_camera_info, self._camera_info_callback, 10
        )
        self.get_logger().info(
            f"Waiting for CameraInfo on {args.topic_camera_info} …"
        )

        # --- Publisher ---
        self.detection_publisher = DetectionPublisher(
            topic_detections='/object_detection/detections',
            topic_image='/object_detection/image',
        )

        self.get_logger().info("ObjectDetectionListener ready.")

    # ------------------------------------------------------------------
    def on_synced_rgb_depth(self, image_msg, depth_msg):
        global last_processed_time
        now = self.get_clock().now().nanoseconds
        wait = self.min_frame_gap_ns + 50_000_000  # add 50 ms buffer
        if (now - last_processed_time) < wait:
            return
        last_processed_time = now
        t0 = self.get_clock().now().nanoseconds
        # Nanoseconds at which the camera captured this frame (from message header)
        capture_ns = image_msg.header.stamp.sec * 1_000_000_000 + \
                     image_msg.header.stamp.nanosec

        if self.intrinsic_matrix is None:
            self.get_logger().warn("CameraInfo not yet received, skipping frame.")
            return

        frame_id = image_msg.header.frame_id

        # --- Convert RGB ---
        try:
            if self.rgb_msg_type is Image:
                rgb = self.bridge.imgmsg_to_cv2(image_msg, desired_encoding="bgr8")
            else:
                rgb = self.bridge.compressed_imgmsg_to_cv2(image_msg)
        except Exception as e:
            self.get_logger().warn(f"RGB conversion failed: {e}")
            return

        # --- Convert depth ---
        try:
            depth = self.bridge.imgmsg_to_cv2(depth_msg).astype(np.float32)
            if depth_msg.encoding == "16UC1":
                depth /= 1000.0          # mm → m
            depth[depth > self.max_depth] = 0
            depth[np.isnan(depth)] = 0
        except Exception as e:
            self.get_logger().warn(f"Depth conversion failed: {e}")
            return

        # Align depth resolution to RGB if they differ.
        if depth.shape[:2] != rgb.shape[:2]:
            depth = depth.reshape((rgb.shape[0], rgb.shape[1]))

        # --- YOLO inference ---
        boxes_xyxy, class_ids, confidences, class_names, masks = self.model_loader.predict(
            rgb, conf=getattr(self.args, 'confidence', 0.5),
            classes=self.target_classes
        )

        if masks is None and len(class_ids) > 0:
            self.get_logger().warn(
                "YOLO returned no masks — make sure you are using a segmentation "
                "model (*-seg.pt), not a detection-only model."
            )

        if len(class_ids) == 0:
            self.get_logger().debug("No detections in this frame.")
            self.detection_publisher.publish(
                rgb, boxes_xyxy, class_ids, confidences, class_names,
                np.empty((0, 3)), frame_id,
                anchors_2d=np.empty((0, 2)),
                masks=None,
                publish_markers=self.publish_markers,
                publish_mask=self.publish_mask,
            )
            self._log_timing(t0, capture_ns)
            return

        # --- Compute 3-D positions ---
        positions_3d, anchors_2d = self._project_boxes_to_3d(boxes_xyxy, depth, masks=masks)

        n = len(class_ids)
        for i in range(n):
            pos = positions_3d[i]
            self.get_logger().info(
                f"  [{class_names[i]}] conf={confidences[i]:.2f} "
                f"pos=({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}) m"
            )

        # --- Publish ---
        self.detection_publisher.publish(
            rgb, boxes_xyxy, class_ids, confidences, class_names,
            positions_3d, frame_id,
            anchors_2d=anchors_2d,
            masks=masks,
            publish_markers=self.publish_markers,
            publish_mask=self.publish_mask,
        )

        self._log_timing(t0, capture_ns)

    # ------------------------------------------------------------------
    def _log_timing(self, t0: int, capture_ns: int):
        """
        Log processing time and end-to-end latency, then update the throttle timer.

        Args:
            t0: nanoseconds at the start of this callback (after throttle check).
            capture_ns: nanoseconds from the image header stamp (camera capture time).
        """
        done_ns = self.get_clock().now().nanoseconds
        processing_ms = (done_ns - t0) / 1e6
        latency_ms = (done_ns - capture_ns) / 1e6
        self.get_logger().info(
            f"  timing — processing: {processing_ms:.1f} ms"
        )
        self.min_frame_gap_ns = int(done_ns - t0)

    # ------------------------------------------------------------------
    def _project_boxes_to_3d(self, boxes_xyxy, depth, masks=None):
        """
        Back-project each detection to 3-D camera coordinates.

        When segmentation masks are available they are used exclusively:
          - depth is sampled only from mask pixels → guaranteed to be on the object
          - the 2-D anchor point is the mask centroid → guaranteed inside the object

        Without masks the centre quarter of the bounding box is used as a
        fallback, which can land on background for irregular shapes.

        Returns:
            positions (N, 3) float array [X, Y, Z] in metres.
            anchors_2d (N, 2) float array [u, v] pixel coords of the anchor point.
            Z == 0 when depth is unavailable.
        """
        K = self.intrinsic_matrix
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]

        h, w = depth.shape[:2]
        positions  = np.zeros((len(boxes_xyxy), 3), dtype=np.float64)
        anchors_2d = np.zeros((len(boxes_xyxy), 2), dtype=np.float64)

        for i, (x1, y1, x2, y2) in enumerate(boxes_xyxy):
            if masks is not None:
                # Resize mask to match depth image resolution
                m = cv2.resize(masks[i], (w, h), interpolation=cv2.INTER_NEAREST)
                mask_pixels = m > 0

                valid_depth = depth[mask_pixels]
                valid_depth = valid_depth[valid_depth > 0]
                if valid_depth.size == 0:
                    continue

                Z = float(np.median(valid_depth))

                # Centroid of the mask as the 2-D anchor point (cv2.moments is faster)
                M = cv2.moments(m)
                if M['m00'] == 0:
                    continue
                u = M['m10'] / M['m00']
                v = M['m01'] / M['m00']
            else:
                # Fallback: centre quarter of the bounding box
                bw, bh = x2 - x1, y2 - y1
                ix1 = int(np.clip(x1 + bw * 0.25, 0, w - 1))
                ix2 = int(np.clip(x2 - bw * 0.25, 0, w - 1))
                iy1 = int(np.clip(y1 + bh * 0.25, 0, h - 1))
                iy2 = int(np.clip(y2 - bh * 0.25, 0, h - 1))
                valid_depth = depth[iy1:iy2, ix1:ix2]
                valid_depth = valid_depth[valid_depth > 0]
                if valid_depth.size == 0:
                    continue
                Z = float(np.median(valid_depth))
                u = (x1 + x2) / 2.0
                v = (y1 + y2) / 2.0

            X = (u - cx) * Z / fx
            Y = (v - cy) * Z / fy
            positions[i]  = [X, Y, Z]
            anchors_2d[i] = [u, v]

        return positions, anchors_2d

    # ------------------------------------------------------------------
    def _camera_info_callback(self, msg):
        self.intrinsic_matrix = np.array(msg.k).reshape(3, 3)
        self._camera_info_received = True
        self.get_logger().info(
            f"CameraInfo received. K={self.intrinsic_matrix}"
        )
        self.destroy_subscription(self._camera_info_sub)
        self._camera_info_sub = None
