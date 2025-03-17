import numpy as np
import open3d as o3d

from sklearn.cluster import KMeans, DBSCAN


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

def edge_detection_2d(points_3D, points_2D, num_slices=100):
    """
    Detect the leftmost and rightmost edges in a 2D point cloud by slicing along the z-axis.

    The function divides the point cloud (projected into 2D as [x, z]) into a number of slices,
    then for each slice finds the point with the minimum x-coordinate (left edge) and the maximum
    x-coordinate (right edge). It also creates a color mapping for visualization purposes.

    Args:
        points_3D (np.ndarray): Nx3 array of 3D points.
        points_2D (np.ndarray): Nx2 array of 2D points with columns [x, z].
        num_slices (int): Number of slices along the z-axis (default is 100).

    Returns:
        tuple: A tuple (points_2D, left_edge_points, right_edge_points) where:
            - points_2D: The original 2D points.
            - left_edge_points: Array of 3D points corresponding to the left edges.
            - right_edge_points: Array of 3D points corresponding to the right edges.
    """
    import numpy as np

    # Determine the z-axis bounds and slice height
    z_min, z_max = np.min(points_2D[:, 1]), np.max(points_2D[:, 1])
    slice_height = (z_max - z_min) / num_slices

    # Precompute slice boundaries and limit to 3/4 of the slices
    slice_bounds = z_min + slice_height * np.arange(num_slices)
    slice_bounds = slice_bounds[: int(num_slices * 3 / 4)]

    left_edge_indices = []
    right_edge_indices = []

    # Process each slice to determine edge indices
    for z_start in slice_bounds:
        z_end = z_start + slice_height
        # Identify points within the current slice along the z-axis
        slice_mask = (points_2D[:, 1] >= z_start) & (points_2D[:, 1] < z_end)
        indices = np.flatnonzero(slice_mask)

        if indices.size > 0:
            slice_points = points_2D[indices]
            # Identify the leftmost (minimum x) and rightmost (maximum x) indices
            left_index = indices[np.argmin(slice_points[:, 0])]
            right_index = indices[np.argmax(slice_points[:, 0])]

            left_edge_indices.append(left_index)
            right_edge_indices.append(right_index)

    # For visualization, create a colors array (default gray; red for left, green for right)
    colors = np.full((points_2D.shape[0], 3), 0.5)
    colors[left_edge_indices] = [1, 0, 0]    # Red for left edges
    colors[right_edge_indices] = [0, 1, 0]     # Green for right edges

    # Extract corresponding 3D points for left and right edges
    left_edge_points = points_3D[left_edge_indices]
    right_edge_points = points_3D[right_edge_indices]

    return left_edge_points, right_edge_points


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


def remove_edge_points(mask, box, depth_image, left_points, right_points, intrinsic_matrix):
    """
    Filters left and right point sets based on valid depth measurements extracted from the mask and depth image.

    Parameters:
        mask (np.ndarray): 2D array with mask values (0 and 255).
        box (tuple/list): Bounding box defined as (left, top, right, bottom).
        depth_image (np.ndarray): 2D depth image.
        left_points (np.ndarray): Array of left edge points (Nx3).
        right_points (np.ndarray): Array of right edge points (Nx3).
        intrinsic_matrix (np.ndarray): Camera intrinsic matrix.

    Returns:
        tuple: Filtered left and right point arrays.
    """
    height, width = depth_image.shape

    def find_last_mask_pixel(col, bottom, top):
        """
        Finds the last row index with a valid mask pixel (255) in the specified column,
        scanning upward from the bottom boundary until the top.
        """
        last_row = None
        for row in range(bottom, top - 1, -1):  # scan upward
            if mask[row, col] == 255:
                last_row = row
            else:
                break  # stop when a non-valid pixel is encountered
        # Fallback to the column index if no valid pixel is found (this behavior is ambiguous and might be revisited)
        return last_row if last_row is not None else col

    # Determine the row indices for left and right boundaries within the box (with slight offset)
    left_col = int(box[0]) + 2
    right_col = int(box[2]) - 2
    top_bound = int(box[1]) + 2
    bottom_bound = int(box[3]) - 2

    left_row_index = find_last_mask_pixel(col=left_col, bottom=bottom_bound, top=top_bound)
    right_row_index = find_last_mask_pixel(col=right_col, bottom=bottom_bound, top=top_bound)

    def get_depth_at(row, col):
        """
        Safely returns the depth value at the specified row and column, clipping indices if necessary.
        """
        row = min(row, height - 1)
        col = min(col, width - 1)
        return depth_image[row, col]

    left_depth = get_depth_at(left_row_index, left_col)
    right_depth = get_depth_at(right_row_index, right_col)

    # Adjust left column if depth value equals 0.5 and row index is within valid range
    if left_depth == 0.5 and left_row_index <= height - 10:
        while left_depth <= 0.5:
            left_col += 5
            left_depth = get_depth_at(left_row_index, left_col)

    def compute_distance_2d(u, v, depth):
        """
        Computes the Euclidean distance of a 3D point derived from pixel (u,v) and its depth.
        """
        fx, fy = intrinsic_matrix[0, 0], intrinsic_matrix[1, 1]
        cx, cy = intrinsic_matrix[0, 2], intrinsic_matrix[1, 2]
        x = (u - cx) * depth / fx
        y = (v - cy) * depth / fy
        z = depth
        # Note: the point is rearranged as (x, z, y)
        return np.linalg.norm((x, z, y))

    def filter_points_by_distance(points, threshold):
        """
        Filters out points that are closer than a given distance threshold.
        """
        if points.size == 0 or points.shape[1] != 3:
            return points
        distances = np.linalg.norm(points, axis=1)
        valid = distances > threshold
        return points[valid]

    left_threshold = compute_distance_2d(u=left_col, v=left_row_index, depth=left_depth)
    right_threshold = compute_distance_2d(u=right_col, v=right_row_index, depth=right_depth)

    filtered_left_points = filter_points_by_distance(left_points, left_threshold)
    filtered_right_points = filter_points_by_distance(right_points, right_threshold)

    # Debug prints to help trace the computation
    '''print(
        "LEFT: U: {}, V: {}, DEPTH: {:.3f}, DIST 2D: {:.3f}, POINTS MIN: {:.3f}, POINTS COUNT: {} -> {}"
        .format(
            left_col,
            left_row_index,
            left_depth,
            left_threshold,
            np.min(np.linalg.norm(filtered_left_points, axis=1)) if filtered_left_points.size else float('nan'),
            len(left_points),
            len(filtered_left_points),
        )
    )
    print(
        "RIGHT: U: {}, V: {}, DEPTH: {:.3f}, DIST 2D: {:.3f}, POINTS MIN: {:.3f}, POINTS COUNT: {} -> {}"
        .format(
            right_col,
            right_row_index,
            right_depth,
            right_threshold,
            np.min(np.linalg.norm(filtered_right_points, axis=1)) if filtered_right_points.size else float('nan'),
            len(right_points),
            len(filtered_right_points),
        )
    )'''

    return filtered_left_points, filtered_right_points
