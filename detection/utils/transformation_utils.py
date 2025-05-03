import open3d as o3d
import numpy as np

def calculate_transformation_matrix(pcd1, pcd2):
    # Faster downsampling
    voxel_size = 0.2  # or even 0.03 if ok
    pcd1_down = pcd1.voxel_down_sample(voxel_size)
    pcd2_down = pcd2.voxel_down_sample(voxel_size)

    # Precomputed normals (faster estimation: use search_param)
    radius_normal = voxel_size * 2
    pcd1_down.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
    pcd2_down.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    # Fast ICP settings
    threshold = voxel_size * 1.5  # matching threshold based on voxel size
    reg_result = o3d.pipelines.registration.registration_icp(
        pcd1_down, pcd2_down, threshold, np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPlane()
    )

    return reg_result.transformation