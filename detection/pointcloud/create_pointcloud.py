import open3d as o3d

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