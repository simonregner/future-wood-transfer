import numpy as np
import open3d as o3d

def pointcloud_to_2d(point_cloud: o3d.geometry.PointCloud) -> np.ndarray:
    """
    Projects a 3D point cloud onto a 2D plane by extracting the X and Z coordinates.

    Args:
        point_cloud (open3d.geometry.PointCloud): Input point cloud with 3D points.

    Returns:
        np.ndarray: A 2D numpy array of shape (N, 2) containing the [X, Z] coordinates.
    """
    points = np.asarray(point_cloud.points)
    return points[:, [0, 2]]