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

    If a file path is provided for the depth image, it is loaded; otherwise, the depth image
    is assumed to be a NumPy array with depth values in meters. The mask (binary, with values 0 or 255)
    is used to filter the depth image. Depth values outside the valid range [0.5, 13] are ignored.

    Args:
        depth_image (str or np.ndarray): Depth image in meters (H, W) or a path to the image file.
        intrinsic_matrix (np.ndarray): Camera intrinsic matrix, shape (3, 3), formatted as:
            [ [fx,  0, cx],
              [ 0, fy, cy],
              [ 0,  0,  1] ]
        mask (np.ndarray): Binary mask (0 or 255) with shape (H, W) to filter the depth image.

    Returns:
        o3d.geometry.PointCloud: Generated 3D point cloud from the masked region.
    """

    # Load depth image if a file path is provided
    if isinstance(depth_image, str):
        depth_image_loaded = cv2.imread(depth_image, cv2.IMREAD_UNCHANGED)
        if depth_image_loaded is None:
            raise ValueError(f"Unable to load depth image from {depth_image}")
        depth_image = depth_image_loaded.astype(np.float32) / 1000.0  # Convert from millimeters to meters
    else:
        depth_image = depth_image.astype(np.float32)

    # Ensure the mask matches the depth image size
    if mask.shape != depth_image.shape:
        mask = cv2.resize(mask, (depth_image.shape[1], depth_image.shape[0]), interpolation=cv2.INTER_NEAREST)

    # Create a valid mask: using the provided mask and excluding zero depth values
    valid_mask = (mask > 0) & (depth_image > 0)

    # Create an array with NaNs and insert valid depth values
    depth_filtered = np.full(depth_image.shape, np.nan, dtype=np.float32)
    depth_filtered[valid_mask] = depth_image[valid_mask]

    # Convert the filtered depth image to an Open3D Image
    depth_o3d = o3d.geometry.Image(depth_filtered)

    # Set up Open3D camera intrinsics
    pinhole_camera_intrinsic = o3d.camera.PinholeCameraIntrinsic()
    pinhole_camera_intrinsic.set_intrinsics(
        width=depth_image.shape[1],
        height=depth_image.shape[0],
        fx=intrinsic_matrix[0, 0],
        fy=intrinsic_matrix[1, 1],
        cx=intrinsic_matrix[0, 2],
        cy=intrinsic_matrix[1, 2]
    )

    # Create and return the point cloud from the depth image
    return o3d.geometry.PointCloud.create_from_depth_image(
        depth_o3d,
        pinhole_camera_intrinsic
    )

