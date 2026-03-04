#!/usr/bin/env python3

import rosbag
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import os
import time
import sys
import re

if len(sys.argv) != 5:
    print("Usage: save_images.py <bag_file> <image_topic> <output_dir> <save_interval>")
    sys.exit(1)

bag_file = sys.argv[1]
image_topic = sys.argv[2]
output_dir = sys.argv[3]
save_interval = float(sys.argv[4])

# Extract bag name without directory and extension
bag_name = os.path.basename(bag_file)
bag_name = re.sub(r'\.bag$', '', bag_name)

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

bridge = CvBridge()
last_save_time = None

with rosbag.Bag(bag_file, 'r') as bag:
    for topic, msg, t in bag.read_messages(topics=[image_topic]):
        msg_time = t.to_sec()
        if last_save_time is None or (msg_time - last_save_time) >= save_interval:
            cv_image = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            filename = os.path.join(output_dir, f"{bag_name}_image_{int(msg_time)}.jpg")
            cv2.imwrite(filename, cv_image)
            print(f"Saved: {filename}")
            last_save_time = msg_time
