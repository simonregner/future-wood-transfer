import open3d as o3d
import numpy as np


def pointcloud_transformation(pointcloud_current, pointcloud_previous):
    """
    Estimate the rigid transformation (rotation + translation) between two consecutive
    point clouds using ICP (Iterative Closest Point) registration.

    Both point clouds are first voxel-downsampled (2 cm grid) and have normals estimated
    to speed up and stabilize the ICP alignment. Registration starts from the identity
    matrix, which works well when motion between frames is small (e.g. sequential camera frames).

    Args:
        pointcloud_current (o3d.geometry.PointCloud): The latest (target) point cloud.
        pointcloud_previous (o3d.geometry.PointCloud): The previous (source) point cloud
                                                       that is aligned onto the current one.

    Returns:
        rotation (np.ndarray): 3x3 rotation matrix from previous to current frame.
        translation (np.ndarray): 3-element translation vector from previous to current frame.
    """
    # Downsample the point clouds to speed up the registration
    voxel_size = 0.02  # adjust voxel size as needed (in meters)
    pointcloud_previous = pointcloud_previous.voxel_down_sample(voxel_size)
    pointcloud_current = pointcloud_current.voxel_down_sample(voxel_size)

    # Estimate normals for better ICP performance (optional but recommended)
    pointcloud_previous.estimate_normals()
    pointcloud_current.estimate_normals()

    # Compute an initial guess.
    # If the images are taken sequentially and the motion is small, the identity matrix may suffice.
    init_trans = np.eye(4)

    # Set the ICP threshold (maximum correspondence distance)
    threshold = 0.05  # adjust threshold as needed

    # Run ICP registration to find the transformation that aligns pcd1 to pcd2
    reg = o3d.pipelines.registration.registration_icp(
        pointcloud_previous, pointcloud_current, threshold, init_trans,
        o3d.pipelines.registration.TransformationEstimationPointToPoint())

    T = reg.transformation

    # Extract rotation and translation
    rotation = T[:3, :3]
    translation = T[:3, 3]

    return rotation, translation