import numpy as np
import open3d as o3d
import cv2

from sklearn.linear_model import RANSACRegressor
from sklearn.cluster import KMeans



def depth_to_pointcloud(depth_image_path, intrinsic_matrix):
    """
    Convert a depth image into a 3D point cloud.

    Args:
        depth_image (numpy.ndarray): Depth image in meters, shape (H, W).
        intrinsic_matrix (numpy.ndarray): Camera intrinsic matrix, shape (3, 3).
                                          [ [fx,  0, cx],
                                            [ 0, fy, cy],
                                            [ 0,  0,  1] ]

    Returns:
        o3d.geometry.PointCloud: Generated 3D point cloud.
    """

    # Load the depth image
    depth_image = cv2.imread(depth_image_path, cv2.IMREAD_UNCHANGED)
    if depth_image is None:
        raise ValueError(f"Unable to load depth image from {depth_image_path}")

    # Convert depth to float32 if not already
    depth_image = depth_image.astype(np.float32)
    print(f"Loaded depth image with shape: {depth_image.shape}")

    # Get image dimensions
    height, width = depth_image.shape

    # Create a meshgrid of pixel indices
    u, v = np.meshgrid(np.arange(width), np.arange(height))

    # Flatten the indices
    u = u.flatten()
    v = v.flatten()

    # Flatten the depth image
    depth = depth_image.flatten()

    # Filter out points with no depth
    valid_depth = depth > 0
    u = u[valid_depth]
    v = v[valid_depth]
    depth = depth[valid_depth]

    # Compute the corresponding 3D points
    fx, fy = intrinsic_matrix[0, 0], intrinsic_matrix[1, 1]
    cx, cy = intrinsic_matrix[0, 2], intrinsic_matrix[1, 2]

    x = (u - cx) * depth / fx
    y = (v - cy) * depth / fy
    z = depth

    # Stack the points into a Nx3 array
    points = np.stack((x, y, z), axis=-1)

    # Create an Open3D point cloud
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)

    return point_cloud


def depth_to_pointcloud_from_mask(depth_image_path, intrinsic_matrix, mask):
    """
    Convert a masked depth image into a 3D point cloud.

    Args:
        depth_image_path (str): Path to the depth image in meters, shape (H, W).
        intrinsic_matrix (numpy.ndarray): Camera intrinsic matrix, shape (3, 3).
                                          [ [fx,  0, cx],
                                            [ 0, fy, cy],
                                            [ 0,  0,  1] ]
        mask (numpy.ndarray): Binary mask (0 or 255) with shape (H, W) to filter depth image.

    Returns:
        o3d.geometry.PointCloud: Generated 3D point cloud from the masked region.
    """

    # Load the depth image
    # image is saved in mm -> need to change back to meter
    depth_image = cv2.imread(depth_image_path, cv2.IMREAD_UNCHANGED) / 1000
    if depth_image is None:
        raise ValueError(f"Unable to load depth image from {depth_image_path}")

    # Convert depth to float32 if not already
    depth_image = depth_image.astype(np.float32)
    print(f"Loaded depth image with shape: {depth_image.shape}")


    # Remove the pixels which are closer than 0.5 meters to the camera
    depth_image[depth_image <= 1] = 0

    # Print the maximum depth value
    max_depth = np.max(depth_image)
    print(f"Maximum depth value: {max_depth} meters")

    # Ensure mask is of type uint8 and has the same size as the depth image
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)
    if mask.shape != depth_image.shape:
        mask = cv2.resize(mask, (depth_image.shape[1], depth_image.shape[0]), interpolation=cv2.INTER_NEAREST)

    # Apply mask to the depth image
    masked_depth_image = cv2.bitwise_and(depth_image, depth_image, mask=mask)

    # Print the maximum depth value
    max_depth = np.max(masked_depth_image)
    print(f"Maximum depth mask value: {max_depth} meters")

    masked_depth_image[masked_depth_image > max_depth * 0.8] = 0

    # Get image dimensions
    height, width = masked_depth_image.shape

    # Create a meshgrid of pixel indices
    u, v = np.meshgrid(np.arange(width), np.arange(height))

    # Flatten the indices
    u = u.flatten()
    v = v.flatten()

    # Flatten the depth image
    depth = masked_depth_image.flatten()

    # Filter out points with no depth (zero or not in mask)
    valid_depth = depth > 0
    u = u[valid_depth]
    v = v[valid_depth]
    depth = depth[valid_depth]

    # Compute the corresponding 3D points
    fx, fy = intrinsic_matrix[0, 0], intrinsic_matrix[1, 1]
    cx, cy = intrinsic_matrix[0, 2], intrinsic_matrix[1, 2]

    x = (u - cx) * depth / fx
    y = (v - cy) * depth / fy
    z = depth

    # Stack the points into a Nx3 array
    points = np.stack((x, y, z), axis=-1)

    # Create an Open3D point cloud
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    point_cloud.paint_uniform_color((0, 0, 0))

    return point_cloud

