import numpy as np


# Fit a line to point cloud
def fit_line(points:np.array):
    x = points[:, 0]
    z = points[:, 2]
    coeffs = np.polyfit(x, z, deg=1)
    return coeffs  # returns a, b


# Intersection computation
def intersection_point(line1, line2):
    a1, b1 = line1
    a2, b2 = line2

    if np.isclose(a1, a2):
        return None  # Parallel

    x_int = (b2 - b1) / (a1 - a2)
    y_int = a1 * x_int + b1
    return np.array([x_int, y_int])


# Check intersection within segment
def segments_intersect(pc1:np.array, pc2:np.array):


    line1 = fit_line(pc1)
    line2 = fit_line(pc2)

    intersect_pt = intersection_point(line1, line2)
    if intersect_pt is None:
        return False, None

    in_seg1 = is_point_in_segment(intersect_pt, pc1)
    in_seg2 = is_point_in_segment(intersect_pt, pc2)

    if in_seg1 and in_seg2:
        return True, intersect_pt
    else:
        return False, intersect_pt


def is_point_in_segment(pt, points, margin=1e-6):
    min_x, max_x = np.min(points[:, 0]), np.max(points[:, 0])
    min_y, max_y = np.min(points[:, 1]), np.max(points[:, 1])
    return (min_x - margin <= pt[0] <= max_x + margin) and \
        (min_y - margin <= pt[1] <= max_y + margin)


def intersection_of_boundaries(boundary_1, boundary_2):
    intersects, _ = segments_intersect(boundary_1, boundary_2)

    return intersects