import numpy as np
import open3d as o3d
import cv2

def create_lines_in_image(image, lines_left, lines_right):
    """
    Project the 3D points of lines_left onto a 2D image using hardcoded camera intrinsics
    and draw them as blue pixels.

    Each 3D point is first transformed into camera coordinates via an identity extrinsic
    matrix (no rotation/translation), then projected using the pinhole model:
        x = fx * X / Z + cx
        y = fy * Y / Z + cy

    Points behind the camera (Z <= 0) or outside the image bounds are skipped.
    The result is displayed with cv2.imshow and also saved to 'projected_image.png'.

    NOTE: lines_right is accepted as a parameter but currently unused.
    NOTE: The camera intrinsics are hardcoded — replace fx, fy, cx, cy with your
          actual calibrated values before use.

    Args:
        image (np.ndarray): BGR image of shape (H, W, 3) to draw onto (modified in-place).
        lines_left (o3d.geometry.PointCloud): 3D point cloud of the left boundary line.
        lines_right (o3d.geometry.PointCloud): 3D point cloud of the right boundary line (unused).
    """
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