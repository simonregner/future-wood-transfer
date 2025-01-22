import rospy

from sensor_msgs.msg import Image
from std_msgs.msg import Header

import numpy as np
import cv2
from cv_bridge import CvBridge


class MaskPublisher:
    def __init__(self,  topic_name='/ml/mask_image'):
        self.publisher = rospy.Publisher(topic_name, Image, queue_size=1)

        # OpenCV Bridge for converting images to ROS messages
        self.bridge = CvBridge()

        self.frame_id = None

    def publish_mask(self, image, results, frame_id):
        """
        Publish an image with overlayed masks.

        Args:
            image (numpy.ndarray): Original image (HxWx3).
            results: Detection results containing masks.
            frame_id (str): ROS frame ID for the published image.
        """
        if results is None:
            # Publish the original image if no results or masks are found
            image_msg = self.bridge.cv2_to_imgmsg(image, encoding="bgr8")
            self.publisher.publish(image_msg)
            return

        self.frame_id = frame_id

        # Precompute dimensions and initialize mask overlay
        image_height, image_width = image.shape[:2]
        mask_overlay = np.zeros_like(image, dtype=np.uint8)

        # Extract masks and process in batches
        masks = results[0].masks.data.cpu().numpy().astype(np.uint8)  # Convert all masks at once
        masks_resized = np.stack([
            cv2.resize(mask, (image_width, image_height), interpolation=cv2.INTER_NEAREST)
            for mask in masks
        ], axis=0)

        # Generate a fixed color (can use random or unique colors if needed)
        color = np.array([255, 0, 0], dtype=np.uint8)

        # Overlay all masks in a vectorized manner
        for scaled_mask in masks_resized:
            mask_colored = (scaled_mask[..., None] * color).astype(np.uint8)  # Broadcast color
            mask_overlay = cv2.addWeighted(mask_overlay, 1.0, mask_colored, 0.5, 0)

        # Combine the overlay with the original image
        result_image = cv2.addWeighted(image, 1.0, mask_overlay, 0.75, 0)

        # Convert to ROS message
        image_msg = self.bridge.cv2_to_imgmsg(result_image, encoding="bgr8")
        image_msg.header = Header()
        image_msg.header.stamp = rospy.Time.now()

        self.publisher.publish(image_msg)