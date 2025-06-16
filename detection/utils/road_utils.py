import sys
sys.path.append("..")

import detection.utils.pointcloud_utils as pointcloud_utils

import rclpy
from rclpy.logging import get_logger

def calculate_road_width(old_road_width, left_paths, right_paths, max_diff_to_old_width=0.3, logger=None):
    """
    Calculate new road width from left/right path arrays.
    If logger is provided (e.g. node.get_logger()), log errors to ROS2 log.
    """

    new_road_width = old_road_width

    for i in range(len(left_paths)):
        if len(left_paths[i]) == 0 or len(right_paths[i]) == 0:
            if logger:
                logger.error(f"Path Pair empty")
            else:
                print("Path Pair empty")
            return None

        new_road_width = (new_road_width + pointcloud_utils.get_distance_between_points(left_paths[i][0], right_paths[i][0])) / 2

    if new_road_width > old_road_width + max_diff_to_old_width:
        new_road_width = old_road_width + max_diff_to_old_width
    elif new_road_width < old_road_width - max_diff_to_old_width:
        new_road_width = old_road_width - max_diff_to_old_width

    return new_road_width
