import sys
sys.path.append("..")

from detection.tools.line_intersection import intersection_of_boundaries

import numpy as np
from itertools import combinations
from scipy.spatial.distance import directed_hausdorff

def find_left_to_right_pairs(boundaries, masks, road_masks, road_width=2.0, logger=None):
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
    """Computes normalized XZ direction vector."""
    if boundary.size < 3:
        return np.zeros(3)

    p1 = np.array([boundary[0][0], boundary[0][2]])
    p2 = np.array([boundary[-1][0], boundary[-1][2]])

    vec = p2 - p1
    norm = np.linalg.norm(vec)
    return (vec / norm) if norm > 0 else np.zeros(3)

def center_point(boundary):
    return np.mean(np.array(boundary), axis=0)

def is_first_left_of_second(boundary_a, boundary_b, direction):
    ca = center_point(boundary_a)
    cb = center_point(boundary_b)
    vec_between = cb - ca
    cross = np.cross(direction, vec_between)
    return cross[2] > 0

def cosine_similarity(a, b):
    if a is None or b is None:
        return 0
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def distance_between_centers(points_a, points_b):
    center_a = np.mean(points_a, axis=0)
    center_b = np.mean(points_b, axis=0)
    return np.linalg.norm(center_a - center_b)

def is_overlap_below_threshold(mask1, mask2, threshold=0.4):
    intersection = np.logical_and(mask1, mask2).sum()
    mask1_area = mask1.sum()
    mask2_area = mask2.sum()
    if mask1_area == 0 or mask2_area == 0:
        return True
    overlap_ratio_1 = intersection / mask1_area
    overlap_ratio_2 = intersection / mask2_area
    return overlap_ratio_1 > 0 or overlap_ratio_2 > 0

def generate_parallel_line(points, parallel_direction=-1, offset_distance=3.0):
    points_xz = points[:, [0, 2]]
    y_values = points[:, 1]
    normals = []
    for i in range(len(points_xz)):
        if i == len(points_xz) - 1:
            direction = points_xz[i] - points_xz[i - 1]
        else:
            direction = points_xz[i + 1] - points_xz[i]
        direction = direction / np.linalg.norm(direction)
        normal = np.array([parallel_direction * direction[1], direction[0]])
        normals.append(normal)
    normals = np.array(normals)
    points_xz_parallel = points_xz + normals * offset_distance
    return np.stack((points_xz_parallel[:, 0], y_values, points_xz_parallel[:, 1]), axis=1)

def normalize_path(path):
    arr = np.array(path)
    centroid = np.mean(arr, axis=0)
    return arr - centroid

def shape_similarity(p1, p2):
    p1n = normalize_path(p1)
    p2n = normalize_path(p2)
    d1 = directed_hausdorff(p1n, p2n)[0]
    d2 = directed_hausdorff(p2n, p1n)[0]
    return max(d1, d2)
