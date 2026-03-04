import pyzed.sl as sl
import bagpy
from bagpy import bagreader
import pandas as pd
import numpy as np
import cv2
import os


rosbag_file = "../../ROSBAG/fwt_2024-06-27-10-18-48_3.bag"
svo_file = "../../SVO/test01.svo"

# Function to convert ROS Image message to a ZED SDK Mat
def ros_img_to_zed_mat(ros_img_path):
    # Read image data from CSV
    img_data = pd.read_csv(ros_img_path)
    if 'height' not in img_data.columns or 'width' not in img_data.columns:
        raise ValueError(f"Invalid image data in file: {ros_img_path}")
    img_np = img_data.values.reshape((img_data['height'][0], img_data['width'][0], -1)).astype(np.uint8)
    # Create ZED Mat from numpy array
    zed_mat = sl.Mat()
    # Allocate memory for ZED Mat with the same resolution as the ROS image
    zed_mat.alloc(sl.Resolution(img_data['width'][0], img_data['height'][0]), sl.MAT_TYPE.U8_C4, sl.MEM.CPU)
    # Convert the ROS image from RGB to RGBA format (adding an alpha channel)
    cv2.cvtColor(img_np, cv2.COLOR_RGB2RGBA, dst=img_np)
    # Set the ZED Mat from the numpy array
    zed_mat.set_from(img_np)
    return zed_mat

# Function to convert ROS Depth Image message to a ZED SDK Mat
def ros_depth_to_zed_mat(depth_img_path):
    # Read depth image data from CSV
    depth_data = pd.read_csv(depth_img_path)
    if 'height' not in depth_data.columns or 'width' not in depth_data.columns:
        raise ValueError(f"Invalid depth data in file: {depth_img_path}")
    depth_np = depth_data.values.reshape((depth_data['height'][0], depth_data['width'][0])).astype(np.uint16)
    # Create ZED Mat from numpy array
    zed_mat = sl.Mat()
    # Allocate memory for ZED Mat with the same resolution as the ROS depth image
    zed_mat.alloc(sl.Resolution(depth_data['width'][0], depth_data['height'][0]), sl.MAT_TYPE.U16_C1, sl.MEM.CPU)
    # Set the ZED Mat from the numpy array
    zed_mat.set_from(depth_np)
    return zed_mat

# Main function to convert ROS bag to SVO file
def main():
    # Define the path to the ROS bag file and the output SVO file
    bagfile_path = rosbag_file  # Replace with your ROS bag file path
    svo_output_path = svo_file  # Replace with the desired .svo file path

    # Create and initialize ZED camera
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD720  # Set camera resolution to HD720
    init_params.depth_mode = sl.DEPTH_MODE.PERFORMANCE  # Enable depth computation with PERFORMANCE mode
    zed = sl.Camera()
    # Open the ZED camera with the specified parameters
    status = zed.open(init_params)
    if status != sl.ERROR_CODE.SUCCESS:
        print(f"Error opening ZED camera: {status}")
        exit(1)

    # Start recording to .svo file
    recording_params = sl.RecordingParameters(svo_output_path, sl.SVO_COMPRESSION_MODE.H264)  # Use H264 compression
    err = zed.enable_recording(recording_params)
    if err != sl.ERROR_CODE.SUCCESS:
        print(f"Error enabling recording: {err}")
        zed.close()
        exit(1)

    # Open the ROS bag file using bagpy
    b = bagreader(bagfile_path)

    # Read messages from specific topics
    rgb_messages = b.message_by_topic('/zed2/zed_node/rgb/image_rect_color')
    depth_messages = b.message_by_topic('/zed2/zed_node/depth/depth_registered')

    # Iterate through RGB messages and depth messages
    if not os.path.exists(rgb_messages):
        print(f"No RGB messages found at {rgb_messages}")
        return
    if not os.path.exists(depth_messages):
        print(f"No Depth messages found at {depth_messages}")
        return

    for msg_path in rgb_messages:
        # Convert the ROS RGB image to a ZED Mat
        try:
            zed_mat = ros_img_to_zed_mat(msg_path)
        except Exception as e:
            print(f"Error processing RGB message: {e}")
            continue
        # Save the current RGB frame to the SVO file
        runtime_parameters = sl.RuntimeParameters()  # Define runtime parameters for grabbing frames
        err = zed.grab(runtime_parameters)
        if err == sl.ERROR_CODE.SUCCESS:
            zed.retrieve_image(zed_mat, sl.VIEW.LEFT)  # Retrieve the left view (RGB image) from the ZED camera

    for msg_path in depth_messages:
        # Convert the ROS depth image to a ZED Mat
        try:
            zed_depth_mat = ros_depth_to_zed_mat(msg_path)
        except Exception as e:
            print(f"Error processing Depth message: {e}")
            continue
        # Save the current depth frame to the SVO file
        runtime_parameters = sl.RuntimeParameters()  # Define runtime parameters for grabbing frames
        err = zed.grab(runtime_parameters)
        if err == sl.ERROR_CODE.SUCCESS:
            zed.retrieve_measure(zed_depth_mat, sl.MEASURE.DEPTH)  # Retrieve the depth measure from the ZED camera

    # Disable recording and close the ZED camera
    zed.disable_recording()
    zed.close()

if __name__ == "__main__":
    main()
