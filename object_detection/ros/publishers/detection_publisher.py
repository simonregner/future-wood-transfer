import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_msgs.msg import Header, ColorRGBA
from geometry_msgs.msg import Point, Vector3
from visualization_msgs.msg import Marker, MarkerArray
from object_detection_msg.msg import DetectedObject, DetectedObjectsMsg


# BGR colors for annotated image
COLORS_BGR = [
    (0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255),
    (255, 255, 0), (255, 0, 255), (128, 255, 0), (0, 128, 255),
]

# Per-class marker appearance: (r, g, b) color and (sx, sy, sz) size in metres
_CLASS_STYLE = {
    'person':     {'color': (1.0, 0.2, 0.2), 'size': (0.5,  1.8, 0.4)},
    'car':        {'color': (0.2, 0.4, 1.0), 'size': (1.8,  1.5, 4.0)},
    'motorcycle': {'color': (1.0, 0.9, 0.1), 'size': (0.8,  1.2, 2.0)},
    'bus':        {'color': (0.2, 0.9, 0.2), 'size': (2.5,  3.0, 10.0)},
    'truck':      {'color': (1.0, 0.5, 0.1), 'size': (2.5,  2.5, 6.0)},
    'bicycle':    {'color': (0.8, 0.2, 0.8), 'size': (0.6,  1.2, 1.8)},
}
_DEFAULT_STYLE = {'color': (0.8, 0.8, 0.8), 'size': (1.0, 1.0, 1.0)}

# Markers expire after this many seconds so stale ones disappear automatically
_MARKER_LIFETIME_SEC = 0.5


class DetectionPublisher(Node):
    """
    Publishes detected objects as:
      - DetectedObjectsMsg  (custom, always published)
      - sensor_msgs/Image   (annotated RGB, always published)
      - sensor_msgs/Image   (segmentation mask overlay, switchable)
      - visualization_msgs/MarkerArray  (RViz 3-D markers, switchable)
    """

    def __init__(
        self,
        topic_detections='/object_detection/detections',
        topic_image='/object_detection/image',
        topic_mask='/object_detection/mask_image',
        topic_markers='/object_detection/markers',
    ):
        super().__init__('detection_publisher')
        self.detections_pub = self.create_publisher(DetectedObjectsMsg, topic_detections, 1)
        self.image_pub = self.create_publisher(Image, topic_image, 1)
        self.mask_pub = self.create_publisher(Image, topic_mask, 1)
        self.markers_pub = self.create_publisher(MarkerArray, topic_markers, 1)
        self.bridge = CvBridge()

    def publish(self, rgb_image, boxes_xyxy, class_ids, confidences, class_names,
                positions_3d, frame_id, anchors_2d=None, masks=None,
                publish_markers=True, publish_mask=True):
        """
        Args:
            rgb_image: BGR numpy array (H, W, 3).
            boxes_xyxy: (N, 4) float array [x1, y1, x2, y2] pixel coords.
            class_ids: (N,) int array.
            confidences: (N,) float array.
            class_names: list of N strings.
            positions_3d: (N, 3) float array [X, Y, Z] metres (camera frame).
            frame_id: ROS frame id string.
            anchors_2d: (N, 2) float array [u, v] pixel coords of the depth anchor.
            masks: (N, H, W) uint8 array of segmentation masks (0/255), or None.
            publish_markers: if False the MarkerArray topic is not published.
            publish_mask: if False the mask image topic is not published.
        """
        stamp = self.get_clock().now().to_msg()
        n = len(class_ids)

        # --- Custom detection message ---
        msg = DetectedObjectsMsg()
        msg.header = Header(stamp=stamp, frame_id=frame_id)
        for i in range(n):
            x1, y1, x2, y2 = boxes_xyxy[i]
            obj = DetectedObject()
            obj.class_name = class_names[i]
            obj.confidence = float(confidences[i])
            obj.bbox = [float(x1), float(y1), float(x2), float(y2)]
            obj.width = float(x2 - x1)
            obj.height = float(y2 - y1)
            pos = positions_3d[i]
            obj.position = Point(x=float(pos[0]), y=float(pos[1]), z=float(pos[2]))
            msg.objects.append(obj)
        self.detections_pub.publish(msg)

        # --- Annotated image ---
        annotated = rgb_image.copy()
        for i in range(n):
            x1, y1, x2, y2 = boxes_xyxy[i].astype(int)
            color = COLORS_BGR[class_ids[i] % len(COLORS_BGR)]
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            label = f"{class_names[i]} {confidences[i]:.2f}"
            dist = float(np.linalg.norm(positions_3d[i]))
            if dist > 0:
                label += f" {dist:.1f}m"
            cv2.putText(annotated, label, (x1 + 4, y1 + 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
            if anchors_2d is not None and len(anchors_2d) > i:
                u, v = int(anchors_2d[i][0]), int(anchors_2d[i][1])
                cv2.circle(annotated, (u, v), 5, color, -1)          # filled dot
                cv2.circle(annotated, (u, v), 6, (255, 255, 255), 1) # white outline
        image_msg = self.bridge.cv2_to_imgmsg(annotated, encoding="bgr8")
        image_msg.header = Header(stamp=stamp, frame_id=frame_id)
        self.image_pub.publish(image_msg)

        # --- Segmentation mask overlay ---
        if publish_mask:
            if masks is not None and len(masks) > 0:
                h, w = rgb_image.shape[:2]
                overlay = np.zeros_like(rgb_image, dtype=np.uint8)
                for i, mask in enumerate(masks):
                    color = COLORS_BGR[class_ids[i] % len(COLORS_BGR)]
                    m = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
                    colored = np.zeros_like(rgb_image, dtype=np.uint8)
                    colored[m > 0] = color
                    overlay = cv2.addWeighted(overlay, 1.0, colored, 1.0, 0)
                mask_image = cv2.addWeighted(rgb_image, 0.6, overlay, 0.4, 0)
            else:
                # No masks available — publish plain RGB so the topic stays active
                mask_image = rgb_image
            mask_msg = self.bridge.cv2_to_imgmsg(mask_image, encoding="bgr8")
            mask_msg.header = Header(stamp=stamp, frame_id=frame_id)
            self.mask_pub.publish(mask_msg)

        # --- RViz MarkerArray ---
        if not publish_markers:
            return

        marker_array = MarkerArray()
        lifetime = Duration(seconds=_MARKER_LIFETIME_SEC).to_msg()

        # Delete all previous markers first
        delete_all = Marker()
        delete_all.header = Header(stamp=stamp, frame_id=frame_id)
        delete_all.action = Marker.DELETEALL
        marker_array.markers.append(delete_all)

        for i in range(n):
            pos = positions_3d[i]
            if pos[2] <= 0:
                continue  # no depth → skip

            style = _CLASS_STYLE.get(class_names[i], _DEFAULT_STYLE)
            r, g, b = style['color']
            sx, sy, sz = style['size']

            # Offset the cube centre so its near face aligns with the detected surface.
            # pos is the front surface point; push the centre backward (away from camera)
            # by half the marker depth along the camera→object direction.
            dist = np.linalg.norm(pos)
            if dist > 0:
                direction = pos / dist
                center = pos + direction * (sz / 2.0)
            else:
                center = pos

            # Cube at 3D position
            cube = Marker()
            cube.header = Header(stamp=stamp, frame_id=frame_id)
            cube.ns = 'object_detection'
            cube.id = i * 2
            cube.type = Marker.CUBE
            cube.action = Marker.ADD
            cube.pose.position = Point(x=float(center[0]), y=float(center[1]), z=float(center[2]))
            cube.pose.orientation.w = 1.0
            cube.scale = Vector3(x=float(sx), y=float(sy), z=float(sz))
            cube.color = ColorRGBA(r=r, g=g, b=b, a=0.4)
            cube.lifetime = lifetime
            marker_array.markers.append(cube)

            # Text label above the cube
            text = Marker()
            text.header = Header(stamp=stamp, frame_id=frame_id)
            text.ns = 'object_detection'
            text.id = i * 2 + 1
            text.type = Marker.TEXT_VIEW_FACING
            text.action = Marker.ADD
            text.pose.position = Point(
                x=float(center[0]),
                y=float(center[1]),
                z=float(center[2]) + sy / 2.0 + 0.2,
            )
            text.pose.orientation.w = 1.0
            text.scale.z = 0.3  # text height in metres
            text.color = ColorRGBA(r=1.0, g=1.0, b=1.0, a=1.0)
            text.text = f"{class_names[i]} {confidences[i]:.2f} ({np.linalg.norm(pos):.1f}m)"
            text.lifetime = lifetime
            marker_array.markers.append(text)

        self.markers_pub.publish(marker_array)
