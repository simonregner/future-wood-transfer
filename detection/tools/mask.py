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
    # Ensure the mask is binary (0 and 255 values)
    _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    # Find all connected components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)

    # Find the largest component (excluding the background, which is label 0)
    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])  # Ignore label 0 (background)

    # Create a mask for the largest component
    largest_component_mask = (labels == largest_label).astype(np.uint8) * 255

    # TODO: Is For testing

    #largest_component_mask = remove_inner_part(largest_component_mask)

    return largest_component_mask