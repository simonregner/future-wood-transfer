import sys
sys.path.append("..")

from detection.tools.line_intersection import intersection_of_boundaries

import numpy as np
from itertools import combinations

from sklearn.cluster import DBSCAN
from scipy.spatial.distance import cdist

import rospy


def direction_vector(boundary:np.array):
    if boundary.size < 3:
        return None

    p1 = np.array([boundary[-3][0], boundary[-3][2]])
    p2 = np.array([boundary[-1][0], boundary[-1][2]])

    p1_mean = np.array([boundary[0][0], boundary[0][2]])
    p2_mean = np.array([boundary[-1][0], boundary[-1][2]])


    #p1, p2 = np.array(boundary[-3]), np.array(boundary[-1])
    #print(p1, p2)
    vec = p2 - p1
    norm = np.linalg.norm(vec)

    vec_mean = p2_mean - p1_mean
    norm_mean = np.linalg.norm(vec_mean)

    vector = vec + vec_mean
    norm_mean = np.linalg.norm(vector)


    return (vec / norm) if norm > 0 else np.zeros(3)

def center_point(boundary):
    return np.mean(np.array(boundary), axis=0)

def angle_similarity(v1, v2):
    return abs(np.dot(v1, v2))

def is_first_left_of_second(boundary_a, boundary_b, direction):
    ca = center_point(boundary_a)
    cb = center_point(boundary_b)
    vec_between = cb - ca
    cross = np.cross(direction, vec_between)
    return cross[2] > 0  # True if A is left of B

def cosine_similarity(a, b):
    if a is None or b is None:
        return None
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def distance_between_centers(points_a, points_b):
    center_a = np.mean(points_a, axis=0)
    center_b = np.mean(points_b, axis=0)
    return np.linalg.norm(center_a - center_b)


def is_overlap_below_threshold(mask1, mask2, threshold=0.4):
    """
    Check if mask1 overlaps mask2 by no more than the given threshold.

    :param mask1: First binary mask as numpy array
    :param mask2: Second binary mask as numpy array
    :param threshold: Maximum allowed overlap percentage (default 0.4 = 40%)
    :return: True if overlap is below threshold, False otherwise
    """
    intersection = np.logical_and(mask1, mask2).sum()
    mask1_area = mask1.sum()
    mask2_area = mask2.sum()

    # Avoid division by zero
    if mask1_area == 0 or mask2_area == 0:
        return True  # No pixels in mask1 to overlap

    overlap_ratio_1 = intersection / mask1_area
    overlap_ratio_2 = intersection / mask2_area

    return overlap_ratio_1 > 0 or overlap_ratio_2 > 0

def generate_parallel_line(points, parallel_direction=-1, offset_distance = 3.0):
    # Step 1: Project to XZ plane
    points_xz = points[:, [0, 2]]  # shape (N, 2)
    y_values = points[:, 1]  # Save Y values

    # Step 2: Compute direction vectors and normals
    normals = []
    for i in range(len(points_xz)):
        if i == len(points_xz) - 1:
            direction = points_xz[i] - points_xz[i - 1]
        else:
            direction = points_xz[i + 1] - points_xz[i]

        direction = direction / np.linalg.norm(direction)  # normalize
        normal = np.array([parallel_direction*direction[1], direction[0]])  # rotate 90° counter-clockwise
        normals.append(normal)

    normals = np.array(normals)

    # Step 3: Offset by 3 meters along the normal direction

    points_xz_parallel = points_xz + normals * offset_distance

    # Step 4: Reconstruct the full 3D points (with original Y)
    return np.stack((points_xz_parallel[:, 0], y_values, points_xz_parallel[:, 1]), axis=1)

def find_left_to_right_pairs(boundaries, masks, road_width=2.0):

    n = len(boundaries)
    vectors = [direction_vector(b) for b in boundaries]

    # Compute all pairwise cosine similarities
    pairs = list(combinations(range(n), 2))

    pair_scores = [[i, j, cosine_similarity(vectors[i], vectors[j]), distance_between_centers(boundaries[i], boundaries[j])] for i, j in pairs]
    pair_scores.sort(key=lambda x: x[2], reverse=True)  # sort by highest similarity

    used = set()
    output_pairs = []

    for i, j, score, distance in pair_scores:
        if intersection_of_boundaries(boundaries[i], boundaries[j]):
            #rospy.logwarn(f"Lane Intersection")
            continue
        if score <= 0.5:
            #rospy.logwarn(f"Score under 0.5: {score}")
            continue
        if is_overlap_below_threshold(masks[i], masks[j], threshold=0.001):
            #rospy.logwarn(f"Overlapping")
            continue
        if i in used or j in used or distance < 1:
            continue
        direction = vectors[i]  # take direction from i (arbitrary)
        if is_first_left_of_second(boundaries[i], boundaries[j], direction):
            output_pairs.append([i, j])  # i is left of j
        else:
            output_pairs.append([j, i])  # j is left of i
        used.update([i, j])

    # Handle unpaired boundary (odd number case)
    unused = [i for i in range(n) if i not in used]
    if unused:
        for idx in unused:
            # Compute global center of pointcloud
            center = [0,0,0]

            boundary_center = boundaries[idx][0]
            side = 'left' if boundary_center[0] < center[0] else 'right'

            if side == 'left':
                # Desired Z offset (positive = up, negative = down)
                z_offset = road_width

                # Create offset in Z direction
                points_shifted = generate_parallel_line(boundaries[idx], 1, offset_distance=road_width)


                N = len(boundaries)

                boundaries_list = [boundaries[i] for i in range(N)]
                # later...
                boundaries_list.append(points_shifted)
                # finally...
                boundaries = np.array(boundaries_list)
                output_pairs.append([idx, N])
            else:
                # Desired Z offset (positive = up, negative = down)
                z_offset = -2.0

                # Create offset in Z direction
                points_shifted = generate_parallel_line(boundaries[idx], -1, offset_distance=road_width)
                N = len(boundaries)

                boundaries_list = [boundaries[i] for i in range(N)]
                # later...
                boundaries_list.append(points_shifted)
                # finally...
                boundaries = np.array(boundaries_list)

                output_pairs.append([N, idx])

    return np.array(output_pairs, dtype=object), boundaries
