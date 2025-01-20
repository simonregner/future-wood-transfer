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


def set_zero_lines(mask):
    #if np.any(mask == 255):
        # Find the row where 255 first appears
        #first_255_row = np.argmin(np.any(mask == 255, axis=1))

        # Set the lines below the first occurrence of 255 to 0
        #mask[:first_255_row + 10] = 0

    return mask

def detect_intersection(mask):
    # Preprocess mask
    smoothed = cv2.GaussianBlur(mask, (5, 5), 0)
    _, binary_mask = cv2.threshold(smoothed, 127, 255, cv2.THRESH_BINARY)

    # Find contours and the intersection center
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)
    M = cv2.moments(largest_contour)
    cx = int(M['m10'] / M['m00'])  # Center x
    cy = int(M['m01'] / M['m00'])  # Center y

    # Create empty masks
    height, width = binary_mask.shape
    masks = {}

    # Find connected components (separate paths)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)

    for i in range(1, num_labels):  # Skip the background
        component_mask = (labels == i).astype(np.uint8) * 255

        # Determine the centroid of the component
        x, y = centroids[i]
        direction = ""

        # Determine the direction of the path relative to the center
        if y < cy and x < cx:
            direction = "top_left"
        elif y < cy and x > cx:
            direction = "top_right"
        elif y > cy and x < cx:
            direction = "bottom_left"
        elif y > cy and x > cx:
            direction = "bottom_right"
        elif y < cy:
            direction = "top"
        elif y > cy:
            direction = "bottom"
        elif x < cx:
            direction = "left"
        elif x > cx:
            direction = "right"

        # Split the component into left and right paths
        if "left" in direction or "right" in direction:
            midpoint = cx if "left" in direction else width - cx
            left_mask = component_mask[:, :midpoint]
            right_mask = component_mask[:, midpoint:]

            masks[f"{direction}_left"] = left_mask
            masks[f"{direction}_right"] = right_mask

        elif "top" in direction or "bottom" in direction:
            midpoint = cy if "top" in direction else height - cy
            left_mask = component_mask[:midpoint, :]
            right_mask = component_mask[midpoint:, :]

            masks[f"{direction}_left"] = left_mask
            masks[f"{direction}_right"] = right_mask

    return masks