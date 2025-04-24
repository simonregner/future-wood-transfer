# Need ROS1 for this code

import rosbag
import cv2
import numpy as np
from cv_bridge import CvBridge
import os
import matplotlib.pyplot as plt

# Parameters
bag_file_path = '/media/simon/T7 Shield/fwt/fwt_2025-04-08-16-08-43_0.bag'  # insert bagfile
rgb_topic = "/hazard_front/zed_node_front/left/image_rect_color"
depth_topic = "/hazard_front/zed_node_front/depth/depth_registered"
camera_topic = "/hazard_front/zed_node_front/depth/camera_info"
output_dir = "/home/simon/Documents/Master-Thesis/ROSBAG_images/MM_INTERSECTION"
frame_skip = 10  # Define how many frames to skip

# Create output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

if not os.path.exists(output_dir + '/images'):
    os.makedirs(output_dir + '/images')

if not os.path.exists(output_dir + '/depth'):
    os.makedirs(output_dir + '/depth')

# Initialize CvBridge
bridge = CvBridge()

# Open the bag file
with rosbag.Bag(bag_file_path, 'r') as bag:
    frame_counter = 0  # Counter to track frames

    rgb_msg = None  # Placeholder for RGB message
    depth_msg = None  # Placeholder for depth message
    camera_msg = None

    for topic, msg, t in bag.read_messages(topics=[rgb_topic, depth_topic, camera_topic]):
        # Store messages from both topics
        if topic == rgb_topic:
            rgb_msg = msg
        elif topic == depth_topic:
            depth_msg = msg
        elif topic == camera_topic:
            camera_msg = msg

        #if camera_msg is not None:
        #    print(camera_msg)


        # Only save if both messages are available and the frame_counter is divisible by frame_skip
        if rgb_msg is not None and depth_msg is not None:
            frame_counter += 1
            if frame_counter % frame_skip == 0:
                # Get the timestamp for unique naming
                timestamp = str(t.to_nsec())

                # Process and save the RGB image
                cv_image = bridge.imgmsg_to_cv2(rgb_msg)
                cv2.imwrite(os.path.join(output_dir, f'images/rgb_{timestamp}.png'), cv_image)
                print(f'Saved RGB image at {timestamp}')

                # Process and save the depth image
                depth_image = bridge.imgmsg_to_cv2(depth_msg)
                #minDepth = 2.0
                #maxDepth = 20.0
                #scale = 255.0 / (maxDepth - minDepth)
                depth_image = np.array(depth_image, dtype=np.float32) * 1000
                #cvimg = np.empty_like(depth_image, dtype=np.uint8)
                #cvimg = cv2.convertScaleAbs(depth_image, alpha=scale, beta=-scale * minDepth)
                #print(depth_image)

                # Replace NaN values with 0 (or another appropriate value)
                depth_array = np.nan_to_num(depth_image, nan=0.0)

                # Clip negative values to 0
                depth_array[depth_array < 0] = 0

                # Clip values to the uint16 range [0, 65535]
                depth_array = np.clip(depth_array, 0, 65535)

                # Convert to uint16
                depth_image_16bit = depth_array.astype(np.uint16)


                cv2.imwrite(os.path.join(output_dir, f'depth/depth_{timestamp}.png'), depth_image_16bit)
                print(f'Saved Depth image at {timestamp}')

                # Reset messages for the next frame
                rgb_msg = None
                depth_msg = None

print('Extraction complete.')