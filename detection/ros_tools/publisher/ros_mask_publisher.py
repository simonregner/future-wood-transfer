import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_msgs.msg import Header

class MaskPublisher(Node):
    def __init__(self, topic_name='/ml/mask_image'):
        super().__init__('mask_publisher')
        self.publisher = self.create_publisher(Image, topic_name, 1)
        self.bridge = CvBridge()
        self.frame_id = None

    def publish_yolo_mask(self, image, mask_boundaries, road_mask, path_pairs, frame_id, yolo_mask=True):
        if mask_boundaries is None or len(path_pairs) == 0:
            image_msg = self.bridge.cv2_to_imgmsg(image, encoding="bgr8")
            image_msg.header = Header()
            image_msg.header.stamp = self.get_clock().now().to_msg()
            image_msg.header.frame_id = frame_id
            self.publisher.publish(image_msg)
            return

        self.frame_id = frame_id
        image_height, image_width = image.shape[:2]
        mask_overlay = np.zeros_like(image, dtype=np.uint8)

        if yolo_mask:
            masks = mask_boundaries[0].masks.data.cpu().numpy().astype(np.uint8)
        else:
            masks = mask_boundaries

        masks_resized = np.stack([
            cv2.resize(mask, (image_width, image_height), interpolation=cv2.INTER_NEAREST)
            for mask in masks
        ], axis=0)

        color = [
            [0, 0, 255], [0, 255, 0], [255, 0, 0], [0, 255, 255], [255, 255, 0], [255, 0, 255],
            [255, 255, 255], [155, 0, 155], [155, 155, 0], [0, 155, 155], [0, 155, 255],
            [0, 255, 155], [255, 155, 0], [155, 255, 0], [155, 0, 255], [255, 0, 155]
        ]

        max_N = len(masks)
        color_counter = 0

        for i, pair in enumerate(path_pairs):
            for idx in pair:
                if idx is None or idx >= max_N:
                    continue
                color_overlay = np.zeros_like(image)
                color_overlay[masks_resized[idx] > 0] = color[i % len(color)]
                mask_overlay = cv2.addWeighted(mask_overlay, 1.0, color_overlay, 1, 0)
                color_counter += 1

        for mask in road_mask:
            color_overlay = np.zeros_like(image)
            color_overlay[mask > 0] = color[color_counter % len(color)]
            mask_overlay = cv2.addWeighted(mask_overlay, 1.0, color_overlay, 0.5, 0)
            color_counter += 1

        result_image = cv2.addWeighted(image, 1.0, mask_overlay, 0.75, 0)
        # scale = 320 / result_image.shape[1] # If you want scaling, implement here

        if yolo_mask:
            image_msg = self.bridge.cv2_to_imgmsg(mask_boundaries[0].plot())
        else:
            image_msg = self.bridge.cv2_to_imgmsg(result_image, encoding="bgr8")

        image_msg.header = Header()
        image_msg.header.stamp = self.get_clock().now().to_msg()
        image_msg.header.frame_id = frame_id
        self.publisher.publish(image_msg)