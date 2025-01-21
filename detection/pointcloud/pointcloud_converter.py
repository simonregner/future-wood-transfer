import numpy as np

def pointcloud_to_2d(point_cloud):
    points = np.asarray(point_cloud.points)
    return points[:, [0, 2]]