import cv2
import numpy as np

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

