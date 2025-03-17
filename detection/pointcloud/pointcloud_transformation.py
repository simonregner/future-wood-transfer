
import open3d as o3d
import numpy as np


def pointcloud_transformation(pointcloud_current, pointcloud_previous):
    # Downsample the point clouds to speed up the registration
    voxel_size = 0.02  # adjust voxel size as needed (in meters)
    pointcloud_previous_down = pointcloud_previous.voxel_down_sample(voxel_size)
    pointcloud_current = pointcloud_current.voxel_down_sample(voxel_size)

    # Estimate normals for better ICP performance (optional but recommended)
    pointcloud_previous_down.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30))
    pointcloud_current.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30))

    # Compute an initial guess.
    # If the images are taken sequentially and the motion is small, the identity matrix may suffice.
    init_trans = np.eye(4)

    # Set the ICP threshold (maximum correspondence distance)
    threshold = 0.05  # adjust threshold as needed

    # Run ICP registration to find the transformation that aligns pcd1 to pcd2
    reg = o3d.pipelines.registration.registration_icp(
        pointcloud_previous_down, pointcloud_current, threshold, init_trans,
        o3d.pipelines.registration.TransformationEstimationPointToPlane())

    T = reg.transformation

    # Extract rotation and translation
    rotation = T[:3, :3]
    translation = T[:3, 3]

    return rotation, translation