import rclpy
from rclpy.serialization import deserialize_message
from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
from rosidl_runtime_py.utilities import get_message
from sensor_msgs.msg import Image
import cv2
import numpy as np


def convert_ros_image_to_cv2(image_msg):
    """
    Converts a ROS2 Image message to an OpenCV format.
    """
    if image_msg.encoding == "rgb8":
        cv2_image = np.frombuffer(image_msg.data, dtype=np.uint8).reshape(image_msg.height, image_msg.width, 3)
    elif image_msg.encoding == "mono8":
        cv2_image = np.frombuffer(image_msg.data, dtype=np.uint8).reshape(image_msg.height, image_msg.width)
    elif image_msg.encoding == "16UC1":  # Depth image encoding
        cv2_image = np.frombuffer(image_msg.data, dtype=np.uint16).reshape(image_msg.height, image_msg.width)
    else:
        raise ValueError(f"Unsupported encoding: {image_msg.encoding}")
    return cv2_image


def extract_images_from_rosbag(bag_path, rgb_topic='/zed2/zed_node/rgb/image_rect_color',
                               depth_topic='/zed2/zed_node/depth/depth_registered'):
    rclpy.init()

    storage_options = StorageOptions(uri=bag_path, storage_id="sqlite3")
    converter_options = ConverterOptions(input_serialization_format="cdr", output_serialization_format="cdr")

    reader = SequentialReader()
    reader.open(storage_options, converter_options)

    image_count = 0
    while reader.has_next():
        topic, data, timestamp = reader.read_next()
        msg_type = get_message(topic)

        # Check if the message is of type Image
        if msg_type == Image and (topic == rgb_topic or topic == depth_topic):
            image_msg = deserialize_message(data, msg_type)
            cv_image = convert_ros_image_to_cv2(image_msg)

            # Define the file name based on topic type and message count
            if topic == rgb_topic:
                filename = f"rgb_image_{image_count:04d}.png"
            elif topic == depth_topic:
                filename = f"depth_image_{image_count:04d}.png"

            # Save the image using OpenCV
            cv2.imwrite(filename, cv_image)
            print(f"Saved {filename}")
            image_count += 1

    rclpy.shutdown()


# Run the function with the path to your bag file
extract_images_from_rosbag('/path/to/your/bagfile')
