import numpy as np
import open3d as o3d
import cv2

def create_lines_in_image(image, lines_left, lines_right):
    # Load the point cloud
    print(image)

    # Get the points and colors from the point cloud
    points = np.asarray(lines_left.points)
    colors = np.asarray(lines_left.colors) * 255  # Convert to 0-255 range

    # Camera intrinsics (example values, replace with your camera parameters)
    fx, fy = 541.736083984375, 541.736083984375  # Focal length
    cx, cy = 642.0556640625, 347.4380187988281  # Principal point
    height, width, channels = image.shape  # Image dimensions

    # Camera extrinsics (example: identity matrix for simplicity)
    extrinsic = np.eye(4)  # Replace with your extrinsic matrix


    # Project points to 2D
    for i, point in enumerate(points):
        # Transform the point to camera coordinates
        point_cam = extrinsic[:3, :3] @ point + extrinsic[:3, 3]

        # Check if the point is in front of the camera
        if point_cam[2] <= 0:
            continue

        # Project the 3D point to 2D
        x = int((fx * point_cam[0] / point_cam[2]) + cx)
        y = int((fy * point_cam[1] / point_cam[2]) + cy)

        # Check if the point is within the image bounds
        if 0 <= x < width and 0 <= y < height:
            image[y, x] = [255,0,0]  # Assign the color to the pixel

    # Save or display the image
    cv2.imshow("Projected Image", image)
    cv2.imwrite("projected_image.png", image)