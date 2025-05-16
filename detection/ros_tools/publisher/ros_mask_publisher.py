import cv2
import numpy as np
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_msgs.msg import Header


class MaskPublisher:
    def __init__(self, topic_name='/ml/mask_image'):
        self.publisher = rospy.Publisher(topic_name, Image, queue_size=1)

        # OpenCV Bridge for converting images to ROS messages
        self.bridge = CvBridge()

        self.frame_id = None

    def publish_yolo_mask(self, image, mask_boundaries, road_mask, path_pairs, frame_id, yolo_mask = True):
        """
        Publish an image with overlayed masks.

        Args:
            image (numpy.ndarray): Original image (HxWx3).
            mask_boundaries: Detection results containing masks.
            frame_id (str): ROS frame ID for the published image.
        """

        if mask_boundaries is None or len(path_pairs) == 0:
            # Publish the original image if no results or masks are found
            image_msg = self.bridge.cv2_to_imgmsg(image, encoding="bgr8")
            self.publisher.publish(image_msg)
            return

        self.frame_id = frame_id

        # Precompute dimensions and initialize mask overlay
        image_height, image_width = image.shape[:2]
        mask_overlay = np.zeros_like(image, dtype=np.uint8)

        # Extract masks and process in batches
        if yolo_mask:
            masks = mask_boundaries[0].masks.data.cpu().numpy().astype(np.uint8)  # Convert all masks at once
        else:
            masks = mask_boundaries

        masks_resized = np.stack([
            cv2.resize(mask, (image_width, image_height), interpolation=cv2.INTER_NEAREST)
            for mask in masks
        ], axis=0)


        # Generate a fixed color (can use random or unique colors if needed)
        color = [[0, 0, 255], [0, 255, 0], [255, 0, 0], [0, 255, 255], [255, 255, 0], [255, 0, 255], [255, 255, 255], [155, 0, 155], [155, 155, 0], [0, 155, 155], [0, 155, 255], [0, 255, 155], [255, 155, 0], [155, 255, 0], [155, 0, 255], [255, 0, 155]]

        # Overlay all masks in a vectorized manner
        #for i, scaled_mask in enumerate(masks_resized):
        #    color_overlay = np.zeros_like(image)
        #    color_overlay[scaled_mask > 0] = color[i]
        #    mask_overlay = cv2.addWeighted(mask_overlay, 1.0, color_overlay, 0.5, 0)

        max_N = len(masks)

        color_counter = 0

        for i, pair in enumerate(path_pairs):
            for idx in pair:
                if idx is None or idx >= max_N:
                    continue

                color_overlay = np.zeros_like(image)
                color_overlay[masks_resized[idx] > 0] = color[i]
                mask_overlay = cv2.addWeighted(mask_overlay, 1.0, color_overlay, 1, 0)

                color_counter += 1

        for mask in road_mask:
            color_overlay = np.zeros_like(image)
            color_overlay[mask > 0] = color[color_counter]
            mask_overlay = cv2.addWeighted(mask_overlay, 1.0, color_overlay, 0.5, 0)
            color_counter += 1

        # Combine the overlay with the original image
        result_image = cv2.addWeighted(image, 1.0, mask_overlay, 0.75, 0)

        scale = 320 / result_image.shape[1]

        #result_image = result_image.reshape((int(result_image.shape[0] * scale), int(result_image.shape[1] * scale), 1))

        # Convert to ROS message
        if yolo_mask:
            image_msg = self.bridge.cv2_to_imgmsg(mask_boundaries[0].plot())
        else:
            image_msg = self.bridge.cv2_to_imgmsg(result_image, encoding="bgr8")
        image_msg.header = Header()
        image_msg.header.stamp = rospy.Time.now()

        self.publisher.publish(image_msg)