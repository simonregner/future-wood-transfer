import numpy as np
import open3d as o3d

def filter_pointcloud_by_distance(points, distance=13) -> np.array:
    distances = np.linalg.norm(points, axis=1)

    # Keep only points within distance
    filtered_points = points[distances <= distance]

    return filtered_points

def get_max_distance_from_pointcloud(points: np.array):
    min_point = np.min(points, axis=0)
    max_point = np.max(points, axis=0)

    return np.linalg.norm(max_point - min_point)

def get_distance_between_points(point_1, point_2):
    return np.linalg.norm(point_1 - point_2)

def create_pointcloud(depth_image, intrinsic_matrix):
    # Convert the filtered depth image to an Open3D Image
    depth_o3d = o3d.geometry.Image(depth_image)

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
