import numpy as np
import open3d as o3d

from sklearn.cluster import KMeans, DBSCAN

from scipy.interpolate import splprep, splev

from scipy.spatial import ConvexHull


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

def edge_detection_2d(points, num_slices=100):
    """
        Detect the leftmost and rightmost edges in a 2D point cloud split along the z-axis.

        Args:
            num_slices (int): Number of slices along the z-axis.
            points (numpy.ndarray): Nx2 array of points with columns [x, z].

        Returns:
            tuple: (all points, left edge points, right edge points)
        """
    z_min, z_max = np.min(points[:, 1]), np.max(points[:, 1])
    slice_height = (z_max - z_min) / num_slices

    # Precompute slice boundaries
    slice_bounds = z_min + slice_height * np.arange(num_slices)
    slice_bounds = slice_bounds[: int(num_slices * 3 / 4)]  # Limit to 3/4 of slices

    # Prepare arrays for indices
    left_edge_indices = []
    right_edge_indices = []

    # Process each slice
    for z_start in slice_bounds:
        z_end = z_start + slice_height

        # Get mask of points in the slice
        slice_mask = (points[:, 1] >= z_start) & (points[:, 1] < z_end)
        indices = np.flatnonzero(slice_mask)  # Get the indices of points in this slice

        if indices.size > 0:
            slice_points = points[indices]

            # Find leftmost and rightmost points by x-coordinate
            left_index = indices[np.argmin(slice_points[:, 0])]
            right_index = indices[np.argmax(slice_points[:, 0])]

            left_edge_indices.append(left_index)
            right_edge_indices.append(right_index)

    # Create colors array for visualization
    colors = np.full((len(points), 3), 0.5)  # Default gray color
    colors[left_edge_indices] = [1, 0, 0]  # Red for left edge
    colors[right_edge_indices] = [0, 1, 0]  # Green for right edge

    # Extract left and right edge points
    left_points = points[left_edge_indices]
    right_points = points[right_edge_indices]

    return points, left_points, right_points


def split_pointcloud(point_cloud):
    points = np.asarray(point_cloud.points)
    if points.size == 0:
        raise ValueError("The point cloud is empty. Please provide a valid point cloud.")

    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(
            f"Expected a point cloud of shape (N, 3), but got shape {points.shape}."
        )
    # Parameters for DBSCAN clustering
    epsilon = 0.5  # Distance threshold
    min_samples = 2  # Minimum points in a cluster

    # Cluster the points
    clustering = DBSCAN(eps=epsilon, min_samples=min_samples).fit(points)
    labels = clustering.labels_

    # Separate clusters
    clusters = {label: points[labels == label] for label in set(labels) if label != -1}

    # Classify clusters as "left" or "right"
    classified_paths = {}
    for label, points in clusters.items():
        centroid = np.mean(points, axis=0)  # Calculate the cluster centroid
        if centroid[0] >= 0:
            classified_paths["right"] = points
        else:
            classified_paths["left"] = points

    if len(set(labels)) < 2:
        return [], [], []

    return point_cloud, classified_paths["left"], classified_paths["right"]



