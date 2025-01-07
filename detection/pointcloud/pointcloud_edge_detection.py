import numpy as np
import open3d as o3d

from sklearn.cluster import KMeans


def edge_detection(point_cloud):
    # Get the updated points after removing outliers
    points = np.asarray(point_cloud.points)

    # Split the point cloud along the longitudinal axis (z-axis)
    num_slices = 50  # Number of slices to split the point cloud into
    z_min, z_max = np.min(points[:, 2]), np.max(points[:, 2])
    slice_height = (z_max - z_min) / num_slices

    left_edge_indices = []
    right_edge_indices = []

    for i in range(num_slices):
        z_start = z_min + i * slice_height
        z_end = z_start + slice_height

        # Get points within the current slice along the z-axis
        slice_mask = (points[:, 2] >= z_start) & (points[:, 2] < z_end)
        slice_points = points[slice_mask]

        if len(slice_points) > 0:
            # Find the leftmost and rightmost points within the slice (based on x-coordinate)
            left_index = np.argmin(slice_points[:, 0])
            right_index = np.argmax(slice_points[:, 0])

            # Get the original indices of these points
            left_edge_indices.append(np.where(slice_mask)[0][left_index])
            right_edge_indices.append(np.where(slice_mask)[0][right_index])

    # Create colors for the point cloud
    colors = np.ones((len(points), 3)) * 0.5  # Default color is gray

    # Color the left points red and right points green
    colors[left_edge_indices] = [1, 0, 0]  # Red color for left side
    colors[right_edge_indices] = [0, 1, 0]  # Green color for right side

    # Assign colors to the point cloud
    point_cloud.colors = o3d.utility.Vector3dVector(colors)

    points = np.asarray(point_cloud.points)

    left_points = points[left_edge_indices]
    right_points = points[right_edge_indices]

    return point_cloud, left_points, right_points

def split_pointcloud(point_cloud):
    points = np.asarray(point_cloud.points)
    if points.size == 0:
        raise ValueError("The point cloud is empty. Please provide a valid point cloud.")

    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(
            f"Expected a point cloud of shape (N, 3), but got shape {points.shape}."
        )
    # Step 1: Calculate the centroid of the point cloud
    centroid = np.mean(points, axis=0)

    # Step 2: Shift the point cloud to make the centroid the origin
    shifted_point_cloud = points - centroid

    # Step 3: Use KMeans clustering to detect two strips
    kmeans = KMeans(n_clusters=2, random_state=0)
    labels = kmeans.fit_predict(shifted_point_cloud[:, :2])  # Use only x and y for clustering

    # Separate the two strips
    strip_1 = points[labels == 0]
    strip_2 = points[labels == 1]

    # Step 4: Determine which is the left and right strip
    strip_1_mean_x = np.mean(strip_1[:, 0])
    strip_2_mean_x = np.mean(strip_2[:, 0])

    if strip_1_mean_x < strip_2_mean_x:
        left_strip, right_strip = strip_1, strip_2
    else:
        left_strip, right_strip = strip_2, strip_1



    return point_cloud, left_strip, right_strip

