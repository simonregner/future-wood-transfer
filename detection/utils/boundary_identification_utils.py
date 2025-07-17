import sys
sys.path.append("..")

from detection.tools.line_intersection import intersection_of_boundaries

import numpy as np
from itertools import combinations
from scipy.spatial.distance import directed_hausdorff

def find_left_to_right_pairs(boundaries, masks, road_masks, road_width=2.0, logger=None):
    """
    Pairs boundaries into left-right pairs based on direction, distance, and shape similarity.
    Generates synthetic boundaries if a boundary has no pair.

    Args:
        boundaries (List[np.ndarray]): List of arrays of 3D points, each representing a boundary.
        masks (List[np.ndarray]): List of binary masks corresponding to each boundary.
        road_masks (List[np.ndarray]): List of masks representing road regions (not used in current logic).
        road_width (float, optional): Offset distance for synthetic boundary generation. Defaults to 2.0.
        logger (Optional[object]): Logger for debug output. If None, no logging is performed.

    Returns:
        Tuple[np.ndarray, np.ndarray]: 
            - Array of index pairs (left, right).
            - Updated boundaries, including synthetic boundaries if added.
    """
    
    n = len(boundaries)
    vectors = [direction_vector(b) for b in boundaries]
    pairs = list(combinations(range(n), 2))

    pair_scores = [[i, j, cosine_similarity(vectors[i], vectors[j]),
                    distance_between_centers(boundaries[i], boundaries[j]),
                    shape_similarity(boundaries[i], boundaries[j])]
                   for i, j in pairs]
    pair_scores.sort(key=lambda x: x[2], reverse=True)  # sort by highest similarity

    used = set()
    output_pairs = []

    for i, j, score, distance, hausdorff in pair_scores:
        maybe_score = score * (3 - hausdorff)
        if (
                score <= 0.3
                or i in used
                or j in used
                or distance < 1
                or distance > 5
                #or is_overlap_below_threshold(masks[i], masks[j], threshold=0.001)
                or intersection_of_boundaries(boundaries[i], boundaries[j])
        ):
            if logger:
                logger.warn(f"Pair skipped: i in used={i in used}, j in used={j in used}, "
                            f"distance < 1: {distance < 1}, distance > 5 {distance > 5}, "
                            f"intersection={intersection_of_boundaries(boundaries[i], boundaries[j])}")
            # or just: print("Pair skipped ...") if no logger provided
            continue

        direction = vectors[i]
        if is_first_left_of_second(boundaries[i], boundaries[j], direction):
            output_pairs.append([i, j])  # i is left of j
        else:
            output_pairs.append([j, i])  # j is left of i
        used.update([i, j])

    # Handle unpaired boundary (odd number case)
    unused = [i for i in range(n) if i not in used]
    if unused:
        for idx in unused:
            center = [0, 0, 0]
            boundary_center = boundaries[idx][0]
            side = 'left' if boundary_center[0] < center[0] else 'right'
            if side == 'left':
                points_shifted = generate_parallel_line(boundaries[idx], 1, offset_distance=road_width)
                N = len(boundaries)
                boundaries_list = [boundaries[i] for i in range(N)]
                boundaries_list.append(points_shifted)
                boundaries = np.array(boundaries_list)
                output_pairs.append([idx, N])
            else:
                points_shifted = generate_parallel_line(boundaries[idx], -1, offset_distance=road_width)
                N = len(boundaries)
                boundaries_list = [boundaries[i] for i in range(N)]
                boundaries_list.append(points_shifted)
                boundaries = np.array(boundaries_list)
                output_pairs.append([N, idx])

    return np.array(output_pairs, dtype=object), boundaries


def direction_vector(boundary: np.array):
    """
    Computes the normalized XZ direction vector from the first to the last point of a boundary.

    Args:
        boundary (np.ndarray): Array of 3D points.

    Returns:
        np.ndarray: Normalized 2D direction vector [x, z]. Returns zeros if norm is zero or too few points.
    """
     
    if boundary.size < 3:
        return np.zeros(3)

    p1 = np.array([boundary[0][0], boundary[0][2]])
    p2 = np.array([boundary[-1][0], boundary[-1][2]])

    vec = p2 - p1
    norm = np.linalg.norm(vec)
    return (vec / norm) if norm > 0 else np.zeros(3)

def center_point(boundary):
    """
    Computes the geometric center (mean point) of a boundary.

    Args:
        boundary (np.ndarray): Array of 3D points.

    Returns:
        np.ndarray: Center point as [x, y, z].
    """

    return np.mean(np.array(boundary), axis=0)

def is_first_left_of_second(boundary_a, boundary_b, direction):
    """
    Determines if boundary_a is to the left of boundary_b relative to a direction vector.

    Args:
        boundary_a (np.ndarray): First boundary points.
        boundary_b (np.ndarray): Second boundary points.
        direction (np.ndarray): Reference direction vector.

    Returns:
        bool: True if boundary_a is to the left of boundary_b, False otherwise.
    """

    ca = center_point(boundary_a)
    cb = center_point(boundary_b)
    vec_between = cb - ca
    cross = np.cross(direction, vec_between)
    return cross[2] > 0

def cosine_similarity(a, b):
    """
    Computes the cosine similarity between two vectors.

    Args:
        a (np.ndarray): First vector.
        b (np.ndarray): Second vector.

    Returns:
        float: Cosine similarity score between -1 and 1.
    """

    if a is None or b is None:
        return 0
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def distance_between_centers(points_a, points_b):
    """
    Computes the Euclidean distance between centers of two sets of points.

    Args:
        points_a (np.ndarray): First set of points.
        points_b (np.ndarray): Second set of points.

    Returns:
        float: Distance between centers.
    """

    center_a = np.mean(points_a, axis=0)
    center_b = np.mean(points_b, axis=0)
    return np.linalg.norm(center_a - center_b)

def is_overlap_below_threshold(mask1, mask2, threshold=0.4):
    """
    Checks if the overlap between two masks exceeds a threshold.

    Args:
        mask1 (np.ndarray): First binary mask.
        mask2 (np.ndarray): Second binary mask.
        threshold (float, optional): Overlap threshold. Defaults to 0.4.

    Returns:
        bool: True if overlap exceeds threshold, False otherwise.
    """

    intersection = np.logical_and(mask1, mask2).sum()
    mask1_area = mask1.sum()
    mask2_area = mask2.sum()
    if mask1_area == 0 or mask2_area == 0:
        return True
    overlap_ratio_1 = intersection / mask1_area
    overlap_ratio_2 = intersection / mask2_area
    return overlap_ratio_1 > 0 or overlap_ratio_2 > 0

def generate_parallel_line(points, parallel_direction=-1, offset_distance=3.0):
    """
    Generates a parallel 3D line offset from the original points (offset in XZ plane).

    Args:
        points (np.ndarray): Array of 3D points (N, 3).
        parallel_direction (int): +1 for right, -1 for left. Defaults to -1.
        offset_distance (float): Offset distance. Defaults to 3.0.

    Returns:
        np.ndarray: New set of points forming the parallel line (N, 3).
    """

    points_xz = points[:, [0, 2]]
    y_values = points[:, 1]
    n_points = len(points_xz)

    directions = np.zeros((n_points - 1, 2))
    normals = np.zeros((n_points, 2))

    # Compute segment directions
    for i in range(n_points - 1):
        vec = points_xz[i + 1] - points_xz[i]
        vec /= np.linalg.norm(vec)
        directions[i] = vec

    # Compute normals (average of adjacent directions)
    for i in range(n_points):
        if i == 0:
            dir_avg = directions[0]
        elif i == n_points - 1:
            dir_avg = directions[-1]
        else:
            dir_avg = directions[i - 1] + directions[i]
            dir_avg /= np.linalg.norm(dir_avg)

        normal = np.array([parallel_direction * dir_avg[1], -parallel_direction * dir_avg[0]])
        normals[i] = normal

    # Apply offset
    points_xz_parallel = points_xz + normals * offset_distance

    # Reconstruct full 3D points
    parallel_points = np.stack((points_xz_parallel[:, 0], y_values, points_xz_parallel[:, 1]), axis=1)

    return parallel_points

def normalize_path(path):
    """
    Normalizes a path by centering it at the origin.

    Args:
        path (np.ndarray): Array of 3D points.

    Returns:
        np.ndarray: Normalized points.
    """

    arr = np.array(path)
    centroid = np.mean(arr, axis=0)
    return arr - centroid

def shape_similarity(p1, p2):
    
    p1n = normalize_path(p1)
    p2n = normalize_path(p2)
    d1 = directed_hausdorff(p1n, p2n)[0]
    d2 = directed_hausdorff(p2n, p1n)[0]
    return max(d1, d2)
