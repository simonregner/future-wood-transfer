"""
bag_reader.py
─────────────
Reads a ROS2 bag (sqlite3 or mcap) and extracts a single synchronized
(RGB, depth, CameraInfo) frame by index.
"""

import os
import numpy as np
import cv2

import rosbag2_py
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import Image, CameraInfo, CompressedImage

# Map ROS2 type strings → message classes
_TYPE_MAP = {
    "sensor_msgs/msg/Image":            Image,
    "sensor_msgs/msg/CompressedImage":  CompressedImage,
    "sensor_msgs/msg/CameraInfo":       CameraInfo,
}

# Map ROS2 image encodings → numpy dtypes
_ENCODING_DTYPE = {
    "bgr8":    (np.uint8,   3),
    "rgb8":    (np.uint8,   3),
    "mono8":   (np.uint8,   1),
    "16UC1":   (np.uint16,  1),
    "32FC1":   (np.float32, 1),
}


def _decode_image(msg) -> np.ndarray:
    """Decode a sensor_msgs/Image to a numpy array without cv_bridge."""
    dtype, channels = _ENCODING_DTYPE.get(msg.encoding, (np.uint8, -1))
    buf = np.frombuffer(bytes(msg.data), dtype=dtype)
    if channels == 1:
        img = buf.reshape(msg.height, msg.width)
    else:
        img = buf.reshape(msg.height, msg.width, channels)
    # Normalize channel order to BGR for downstream cv2 compatibility
    if msg.encoding == "rgb8":
        img = img[:, :, ::-1].copy()
    return img


class BagFrameReader:
    def __init__(self, config):
        self.bag_path          = config.rosbag_path
        self.topic_rgb         = config.topic_rgb
        self.topic_depth       = config.topic_depth
        self.topic_camera_info = config.topic_camera_info
        self.rgb_image_type    = getattr(config, "rgb_image_type", "Image")

    # ──────────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _resolve_bag_uri(self):
        """
        Expand ~ and resolve the URI rosbag2_py expects.
        rosbag2_py wants the bag *directory* (which contains metadata.yaml),
        not a .db3 / .mcap file directly.
        """
        path = os.path.expanduser(self.bag_path)
        # If the user pointed at a .db3 or .mcap file, use its parent directory
        if os.path.isfile(path) and path.endswith((".db3", ".mcap")):
            path = os.path.dirname(path)
        return path

    def _detect_storage_id(self, bag_dir: str):
        for fname in os.listdir(bag_dir):
            if fname.endswith(".mcap"):
                return "mcap"
        return "sqlite3"

    def _read_all_messages(self):
        """Stream the bag and bucket messages by topic."""
        bag_dir = self._resolve_bag_uri()
        print(f"[BagReader] Opening bag directory: {bag_dir}")
        storage_options = rosbag2_py.StorageOptions(
            uri=bag_dir,
            storage_id=self._detect_storage_id(bag_dir),
        )
        converter_options = rosbag2_py.ConverterOptions("", "")

        reader = rosbag2_py.SequentialReader()
        reader.open(storage_options, converter_options)

        type_map = {t.name: t.type for t in reader.get_all_topics_and_types()}

        topics_needed = {self.topic_rgb, self.topic_depth, self.topic_camera_info}
        messages = {t: [] for t in topics_needed}

        while reader.has_next():
            topic, data, timestamp = reader.read_next()
            if topic not in topics_needed:
                continue
            msg_cls = _TYPE_MAP.get(type_map.get(topic, ""))
            if msg_cls is None:
                continue
            messages[topic].append((timestamp, deserialize_message(data, msg_cls)))

        return messages

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    def count_frames(self):
        """Return the number of available RGB frames in the bag."""
        messages = self._read_all_messages()
        return len(messages[self.topic_rgb])

    def get_frame(self, frame_index: int):
        """
        Extract a specific frame by index.

        Returns
        -------
        rgb_image        : np.ndarray (H, W, 3)  BGR, uint8
        depth_image      : np.ndarray (H, W)     float32, metres
        intrinsic_matrix : np.ndarray (3, 3)     pinhole K matrix
        """
        messages = self._read_all_messages()

        rgb_msgs        = messages[self.topic_rgb]
        depth_msgs      = messages[self.topic_depth]
        camera_info_msgs = messages[self.topic_camera_info]

        if not rgb_msgs:
            raise ValueError(f"No RGB messages found on topic '{self.topic_rgb}'")
        if not depth_msgs:
            raise ValueError(f"No depth messages found on topic '{self.topic_depth}'")
        if frame_index >= len(rgb_msgs):
            raise ValueError(
                f"frame_index={frame_index} out of range "
                f"(bag has {len(rgb_msgs)} RGB frames)"
            )

        rgb_ts, rgb_msg = rgb_msgs[frame_index]

        # Closest depth frame
        depth_ts, depth_msg = min(depth_msgs, key=lambda x: abs(x[0] - rgb_ts))

        # Closest camera_info (mandatory)
        if not camera_info_msgs:
            raise ValueError(f"No CameraInfo messages on topic '{self.topic_camera_info}'")
        _, info_msg = min(camera_info_msgs, key=lambda x: abs(x[0] - rgb_ts))

        # ── decode RGB ────────────────────────────────────────────────────────
        if self.rgb_image_type == "CompressedImage":
            buf = np.frombuffer(bytes(rgb_msg.data), np.uint8)
            rgb_image = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        else:
            rgb_image = _decode_image(rgb_msg)   # → BGR uint8

        # ── decode depth ──────────────────────────────────────────────────────
        depth_image = _decode_image(depth_msg)
        print(f"[BagReader] Depth encoding={depth_msg.encoding}  raw dtype={depth_image.dtype}  "
              f"raw range=[{depth_image[depth_image > 0].min():.1f}, {depth_image.max():.1f}]")
        if depth_image.dtype == np.uint16:
            depth_image = depth_image.astype(np.float32) / 1000.0   # mm → m
        else:
            depth_image = depth_image.astype(np.float32)
        valid = depth_image > 0
        print(f"[BagReader] Depth in metres: min={depth_image[valid].min():.2f}  "
              f"max={depth_image[valid].max():.2f}  mean={depth_image[valid].mean():.2f}")

        # ── intrinsics ────────────────────────────────────────────────────────
        intrinsic_matrix = np.array(info_msg.k, dtype=np.float64).reshape(3, 3)

        dt_ms = abs(depth_ts - rgb_ts) / 1e6
        print(f"[BagReader] frame {frame_index}  |  "
              f"rgb_ts={rgb_ts}  depth_dt={dt_ms:.1f} ms")
        print(f"[BagReader] RGB   {rgb_image.shape}  "
              f"Depth {depth_image.shape}  "
              f"fx={intrinsic_matrix[0,0]:.1f}  cx={intrinsic_matrix[0,2]:.1f}")

        return rgb_image, depth_image, intrinsic_matrix
