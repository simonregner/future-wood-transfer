import numpy as np
import open3d as o3d
import cv2



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


def depth_to_pointcloud_from_mask(depth_image, intrinsic_matrix, mask):
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

    # Load depth image if a path is provided
    if isinstance(depth_image, str):
        depth_image = cv2.imread(depth_image, cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000
        if depth_image is None:
            raise ValueError(f"Unable to load depth image from {depth_image}")
    else:
        depth_image = depth_image.astype(np.float32)

    # Remove pixels closer than 0.5 meters
    depth_image[depth_image <= 0.5] = 0

    # Ensure mask matches depth image size
    if mask.shape != depth_image.shape:
        mask = cv2.resize(mask, (depth_image.shape[1], depth_image.shape[0]), interpolation=cv2.INTER_NEAREST)

    # Combine mask with depth validity
    valid_mask = (mask > 0) & (depth_image > 0)

    # Extract valid depth values
    depth = depth_image[valid_mask]

    if depth.size == 0:  # If no valid points, return an empty point cloud
        return o3d.geometry.PointCloud()

    # Get the pixel coordinates of valid points
    v, u = np.where(valid_mask)  # v = row (y-coord), u = col (x-coord)

    # Extract intrinsic matrix parameters
    fx, fy = intrinsic_matrix[0, 0], intrinsic_matrix[1, 1]
    cx, cy = intrinsic_matrix[0, 2], intrinsic_matrix[1, 2]

    # Convert pixel coordinates and depth to 3D coordinates
    x = (u - cx) * depth / fx
    y = (v - cy) * depth / fy
    z = depth

    # Stack into an Nx3 array
    points = np.column_stack((x, y, z))

    # Create Open3D point cloud
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)

    return point_cloud

