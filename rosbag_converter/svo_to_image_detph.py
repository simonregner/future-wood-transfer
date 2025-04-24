import pyzed.sl as sl
import cv2
import numpy as np
import os

# Parameters
svo_folder = "/media/simon/T7 Shield/FutureWoodTransfer/01_1"  # Folder with .svo2 files
image_output_folder = "../../SVO/MM_ForestRoads_01/images"
depth_output_folder = "../../SVO/MM_ForestRoads_01/depth"
N_ms = 500  # Save images every N milliseconds

# Desired resolution
TARGET_WIDTH = 1280
TARGET_HEIGHT = 720

# Ensure output folders exist
os.makedirs(image_output_folder, exist_ok=True)
os.makedirs(depth_output_folder, exist_ok=True)

# Go through all .svo2 files in the folder
for filename in os.listdir(svo_folder):
    if filename.endswith(".svo2"):
        svo_file = os.path.join(svo_folder, filename)
        svo_base_name = os.path.splitext(filename)[0]

        print(f"\nProcessing: {svo_file}")

        # Initialize ZED
        zed = sl.Camera()

        init_params = sl.InitParameters()
        init_params.set_from_svo_file(svo_file)
        init_params.svo_real_time_mode = False
        init_params.camera_resolution = sl.RESOLUTION.HD720
        init_params.depth_mode = sl.DEPTH_MODE.ULTRA

        status = zed.open(init_params)
        if status != sl.ERROR_CODE.SUCCESS:
            print("Error opening SVO:", status)
            continue

        image = sl.Mat()
        depth = sl.Mat()
        runtime_params = sl.RuntimeParameters()

        start_timestamp = zed.get_timestamp(sl.TIME_REFERENCE.CURRENT).get_milliseconds()
        next_save_timestamp = start_timestamp + N_ms
        frame_count = 0

        while zed.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:
            current_timestamp = zed.get_timestamp(sl.TIME_REFERENCE.CURRENT).get_milliseconds()

            if current_timestamp >= next_save_timestamp:
                zed.retrieve_image(image, sl.VIEW.LEFT)
                zed.retrieve_measure(depth, sl.MEASURE.DEPTH)

                image_np = image.get_data()[:, :, :3].copy()
                depth_np = depth.get_data()
                depth_np_cleaned = np.nan_to_num(depth_np, nan=0.0, posinf=0.0, neginf=0.0)

                # Save left image
                image_path = os.path.join(image_output_folder, f"{svo_base_name}_left_image_{frame_count}.png")
                cv2.imwrite(image_path, image_np)

                # Normalize and save depth image
                depth_image_normalized = cv2.normalize(depth_np_cleaned, None, 0, 255, cv2.NORM_MINMAX)
                depth_image_normalized = np.uint8(depth_image_normalized)
                depth_path = os.path.join(depth_output_folder, f"{svo_base_name}_depth_image_{frame_count}.png")
                cv2.imwrite(depth_path, depth_image_normalized)

                print(f"  Saved frame {frame_count} at {current_timestamp - start_timestamp} ms")
                next_save_timestamp += N_ms
                frame_count += 1

        zed.close()
        print(f"Finished processing {svo_file}")
