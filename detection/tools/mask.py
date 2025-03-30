import sys
sys.path.append("..")

import cv2
import numpy as np

import detection.tools.skeleton as sk

from scipy.spatial import KDTree
import networkx as nx


def remove_inner_part(mask):
    # Define the kernel size for erosion (5 pixels)
    kernel_size = 2
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    # Erode the mask
    eroded_mask = cv2.erode(mask, kernel, iterations=1)

    # Subtract the eroded mask from the original mask to get the boundary
    boundary = cv2.subtract(mask, eroded_mask)

    return boundary

def keep_largest_component(mask):
    """
    Keep only the largest connected component in a binary mask.

    Parameters:
        mask (numpy.ndarray): Input binary mask (0 and 255 values).

    Returns:
        numpy.ndarray: Binary mask with only the largest connected component.
    """
    # Ensure the mask is binary
    binary_mask = (mask > 127).astype(np.uint8)

    # Find connected components and their stats
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)

    if num_labels <= 1:  # Only background found
        return np.zeros_like(mask, dtype=np.uint8)

    # Find the largest component (excluding the background, which is label 0)
    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])  # Skip background label

    # Directly generate the output mask
    return (labels == largest_label).astype(np.uint8) * 255



def sliding_window_lane_detection(binary_mask, side="left", window_width=50, min_pixels=50):
    """
    Extract lane masks using the sliding window approach.

    :param binary_mask: Binary input mask (numpy array).
    :param side: Side of the lane to detect ('left' or 'right').
    :param window_width: Width of the sliding window.
    :param min_pixels: Minimum number of pixels to detect as part of a lane.
    :return: Lane mask (numpy array).
    """
    # Get dimensions of the binary mask
    height, width = binary_mask.shape

    # Determine x_start based on the side
    if side == "left":
        x_start = width // 4
        x_end = width // 2
    else:
        x_start = width // 2
        x_end = 3 * width // 4

    # Initialize lane mask and window centers
    lane_mask = np.zeros_like(binary_mask)
    window_centers = []

    # Create histogram to find starting points
    histogram = np.sum(binary_mask[height // 2:, x_start:x_end], axis=0)
    x_base = np.argmax(histogram) + x_start

    # Sliding window parameters
    window_height = height // 20  # Height of each window
    x_current = x_base

    for window in range(20):  # Divide image into 20 horizontal slices
        y_low = height - (window + 1) * window_height
        y_high = height - window * window_height

        # Define the window boundaries
        x_low = max(0, x_current - window_width)
        x_high = min(width, x_current + window_width)

        # Extract window region
        window_region = binary_mask[y_low:y_high, x_low:x_high]

        # Find nonzero pixels within the window
        nonzero = cv2.findNonZero(window_region)
        if nonzero is not None and len(nonzero) > min_pixels:
            # Update the x-coordinate of the window center
            x_current = int(np.mean(nonzero[:, 0, 0]) + x_low)
            window_centers.append((x_current, (y_low + y_high) // 2))

            # Update lane mask with detected pixels
            lane_mask[y_low:y_high, x_low:x_high] = 255

    return lane_mask

def extract_lane_masks(road_mask):
    """
    Extract the left and right lane masks from a road mask using sliding window detection.

    :param road_mask: Input binary mask of the road (numpy array).
    :return: Tuple of left_lane_mask and right_lane_mask.
    """
    # Extract left lane
    left_lane_mask = sliding_window_lane_detection(road_mask, side="left")

    # Extract right lane
    right_lane_mask = sliding_window_lane_detection(road_mask, side="right")

    return left_lane_mask, right_lane_mask

def get_mask_edge_distance(mask):
    # For the left side
    left_condition = mask[:, 0] != 255
    # np.where returns all indices where condition is True
    left_indices = np.where(left_condition)[0]
    # Get the bottom-most one (largest index)
    if left_indices.size > 0:
        left_row_index = left_indices[-1]
    else:
        left_row_index = None

    # For the right side
    right_condition = mask[:, -1] != 255
    right_indices = np.where(right_condition)[0]
    if right_indices.size > 0:
        right_row_index = right_indices[-1]
    else:
        right_row_index = None

    return left_row_index, right_row_index


'''def process_mask_to_left_right_masks(mask, width=10):
    # Step 1: Extract largest contour
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = max(contours, key=cv2.contourArea)

    # Step 2: Skeletonize and find main skeleton
    skeleton = sk.get_skeleton(mask)
    graph = sk.skeleton_to_graph(skeleton)
    centerline = sk.longest_path_in_graph(graph)

    # Step 3: Compute normals
    normals = sk.get_normals(centerline)

    # Step 3: Create left/right masks of specified width
    mask_left, mask_right = create_side_masks(mask.shape, centerline, normals, width)

    return mask_left, mask_right, contour, centerline
    '''

def get_bottom_center_point(mask, box_value):
    y_max = box_value[3]
    ### STEP 1: Find the actual bottom-center of the road using the mask ###
    bottom_y = int(y_max)  # Start from the height of the mask
    while bottom_y > 0 and np.sum(mask[bottom_y, :] == 255) == 0:
        bottom_y -= 1  # Move upwards until we find road pixels

    # Get all 255-pixels at this bottom line
    bottom_x_indices = np.where(mask[bottom_y, :] == 255)[0]

    if len(bottom_x_indices) == 0:
        raise ValueError("No road pixels (255) found at the detected mask bottom.")

    # Compute center-bottom of road from 255-pixel values
    bottom_center_x = int(np.mean(bottom_x_indices))
    return np.array([bottom_y, bottom_center_x], dtype=int)

def create_side_masks_from_mask(mask, box_values, width=10):
    # Skeletonize and find main centerline


    skeleton = sk.get_skeleton(mask)
    graph = sk.skeleton_to_graph(skeleton)

    #graph = sk.remove_bottom_corner_nodes(graph, mask)
    '''
    car_point = get_bottom_center_point(mask, box_values)

    nearest_node_idx, nearest_node = sk.get_nearest_node(graph, car_point)
    graph = sk.remove_nearest_nodes_ends(graph, nearest_node)


    graph, start_node_idx = sk.add_start_node(graph, car_point, nearest_node_idx)

    longest_path_nodes = sk.find_longest_path_from_start(graph, start_node_idx)

    # Extract the edges that belong to the longest path
    longest_path_edges = [(longest_path_nodes[i], longest_path_nodes[i + 1]) for i in
                          range(len(longest_path_nodes) - 1)]

    ### STEP 4: Create a new graph containing only the longest path ###
    longest_graph = nx.Graph()
    for node in longest_path_nodes:
        longest_graph.add_node(node, o=graph.nodes[node]['o'])

    for edge in longest_path_edges:
        s, e = edge
        longest_graph.add_edge(s, e, weight=graph[s][e]['weight'], pts=graph[s][e]['pts'])

    graph = sk.extend_path(graph, mask, longest_path_nodes)
    '''

    '''normals = sk.get_normals(longest_path_nodes)

    print(longest_path_nodes)

    # Create KDTree for efficient nearest-centerline lookup
    tree = KDTree(longest_path_nodes)

    # Find all road pixels in mask
    road_pixels = np.column_stack(np.where(mask > 0))
    road_pixels_xy = road_pixels[:, ::-1]  # (y,x) -> (x,y)

    # Find nearest points on centerline
    distances, indices = tree.query(road_pixels_xy)
    nearest_center_pts = longest_path_nodes[indices]

    # Compute vectors from centerline to road pixels
    vectors = road_pixels_xy - nearest_center_pts

    # Compute side classification
    sides = np.einsum('ij,ij->i', vectors, normals[indices])
'''
    mask_left = np.zeros_like(mask, dtype=np.uint8)
    mask_right = np.zeros_like(mask, dtype=np.uint8)

    '''for i, (y, x) in enumerate(road_pixels):
        dist_to_edge = np.linalg.norm(road_pixels_xy[i] - longest_path_nodes[indices[i]])
        if dist_to_edge <= width:
            if sides[i] > 0:
                mask_left[y, x] = 255
            else:
                mask_right[y, x] = 255
'''
    return mask_left, mask_right, graph #sk.longest_graph(graph, start_node_idx)

def reduce_mask_width(mask, bbox):
    # Set threshold values
    max_width_n = 800  # Maximum allowed road width at the bottom
    reduce_height_y = 300  # Height in pixels where width reduction applies

    ### STEP 1: Set bottom_y from Bounding Box ###
    bottom_y = int(bbox[3])  # Ensure bottom_y is an integer

    # Find leftmost and rightmost pixels of the road at bottom_y
    if 0 <= bottom_y < mask.shape[0]:  # Ensure `bottom_y` is within valid range
        road_pixels = np.where(mask[bottom_y, :] == 255)[0]
        if len(road_pixels) > 0:
            left_edge = road_pixels[0]
            right_edge = road_pixels[-1]
            road_width = right_edge - left_edge
        else:
            raise ValueError(f"No road pixels found at bottom_y ({bottom_y}).")
    else:
        raise ValueError(f"bottom_y ({bottom_y}) is out of bounds for mask height {mask.shape[0]}.")

    ### STEP 2: Reduce Width from Left & Right Edges (Preserve Center) ###
    if road_width > max_width_n:
        print(f"Road width at bottom ({road_width}px) is wider than {max_width_n}px. Reducing edges...")

        # Define region for reduction (only reduce the bottom `y` pixels)
        bottom_mask = mask[max(0, bottom_y - reduce_height_y):bottom_y + 1, :].copy()

        # Calculate how much to reduce from both sides
        excess_width = road_width - max_width_n
        pixels_to_remove_per_side = excess_width // 2  # Reduce equally from both sides

        # Remove pixels from left and right edges while preserving the center
        for y in range(bottom_mask.shape[0]):
            left_limit = max(0, left_edge + pixels_to_remove_per_side)
            right_limit = min(mask.shape[1], right_edge - pixels_to_remove_per_side)
            bottom_mask[y, :left_limit] = 0  # Remove from left
            bottom_mask[y, right_limit:] = 0  # Remove from right

        # Merge back into the original mask
        mask[max(0, bottom_y - reduce_height_y):bottom_y + 1, :] = bottom_mask

    return mask


