import numpy as np
import cv2

def mask_to_contour(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Pick the largest contour assuming it's the road
    largest_contour = max(contours, key=cv2.contourArea)

    return largest_contour

def classify_contour_points(contour, centerline, normals):
    from scipy.spatial import KDTree
    tree = KDTree(centerline)
    left_pine, right_pine = [], []

    for point in contour.reshape(-1, 2):
        dist, idx = tree.query(point)
        center_pt = centerline[idx]
        normal = normals[idx]
        vec = point - center_pt
        side = np.dot(vec, normal)

        if side > 0:
            left_pine.append(point)
        else:
            right_pine.append(point)

    return np.array(left_pine), np.array(right_pine)