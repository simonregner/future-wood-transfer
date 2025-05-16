import numpy as np

# Fit a polynomial of degree n (not just a line)
def fit_polynomial(points: np.array, degree: int = 3):
    x = points[:, 0]
    z = points[:, 2]

    if len(np.unique(x)) <= degree:
        raise ValueError("Not enough unique x-values to fit polynomial of this degree.")

    coeffs = np.polyfit(x, z, deg=degree)
    return coeffs  # returns highest degree first: [a_n, ..., a_1, a_0]


# Evaluate polynomial at given x
def eval_polynomial(coeffs, x):
    return np.polyval(coeffs, x)


# Find intersection of two polynomial curves
def intersection_point(poly1, poly2, x_range, num_points=1000):
    xs = np.linspace(x_range[0], x_range[1], num_points)
    y1 = eval_polynomial(poly1, xs)
    y2 = eval_polynomial(poly2, xs)

    diff = np.abs(y1 - y2)
    idx = np.argmin(diff)

    if diff[idx] > 1e-3:  # no good intersection found
        return None

    x_int = xs[idx]
    y_int = y1[idx]
    return np.array([x_int, y_int])


def is_point_in_segment(pt, points, margin=1e-6):
    min_x, max_x = np.min(points[:, 0]), np.max(points[:, 0])
    min_z, max_z = np.min(points[:, 2]), np.max(points[:, 2])
    return (min_x - margin <= pt[0] <= max_x + margin) and \
           (min_z - margin <= pt[1] <= max_z + margin)


def segments_intersect(pc1: np.array, pc2: np.array, degree=2):
    poly1 = fit_polynomial(pc1, degree)
    poly2 = fit_polynomial(pc2, degree)

    x_range = (
        max(np.min(pc1[:, 0]), np.min(pc2[:, 0])),
        min(np.max(pc1[:, 0]), np.max(pc2[:, 0]))
    )

    if x_range[0] >= x_range[1]:
        return False, None  # No overlap in x-range

    intersect_pt = intersection_point(poly1, poly2, x_range)

    if intersect_pt is None:
        return False, None

    in_seg1 = is_point_in_segment(intersect_pt, pc1)
    in_seg2 = is_point_in_segment(intersect_pt, pc2)

    if in_seg1 and in_seg2:
        return True, intersect_pt
    else:
        return False, intersect_pt


def intersection_of_boundaries(boundary_1, boundary_2, degree=2):
    intersects, _ = segments_intersect(boundary_1, boundary_2, degree=degree)
    return intersects