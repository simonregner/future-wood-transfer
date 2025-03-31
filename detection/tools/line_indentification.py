import numpy as np
from itertools import combinations

def direction_vector(boundary):
    if len(boundary) < 2:
        return np.zeros(3)
    p1, p2 = np.array(boundary[-2]), np.array(boundary[-1])
    vec = p2 - p1
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else np.zeros(3)

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
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def find_left_to_right_pairs(boundaries):

    n = len(boundaries)
    vectors = [direction_vector(b) for b in boundaries]

    # Compute all pairwise cosine similarities
    pairs = list(combinations(range(n), 2))

    print(pairs)

    pair_scores = [[i, j, cosine_similarity(vectors[i], vectors[j])] for i, j in pairs]
    pair_scores.sort(key=lambda x: x[2], reverse=True)  # sort by highest similarity

    print(pair_scores)

    used = set()
    output_pairs = []

    for i, j, _ in pair_scores:
        if i in used or j in used:
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
        idx = unused[0]
        # Compute global center of pointcloud
        all_points = np.vstack(boundaries)
        center = np.mean(all_points, axis=0)

        boundary_center = center_point(boundaries[idx])
        side = 'left' if boundary_center[0] < center[0] else 'right'

        if side == 'left':
            output_pairs.append([idx, None])
        else:
            output_pairs.append([None, idx])

    return np.array(output_pairs, dtype=object)
