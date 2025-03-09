import cv2
import numpy as np
from ultralytics import YOLO  # YOLO model
import cv2.ximgproc as xip  # For skeleton thinning
from skimage.morphology import skeletonize

# Set image and model paths
image_path = "2022-10-12_solalinden_waldwege__0095_1665579514737884132_windshield_vis.png"
model = YOLO("../yolo_V11/runs/segment/train_seperate_road_intersection/weights/best.pt")

# Load the image
img = cv2.imread(image_path)
if img is None:
    raise ValueError(f"Image not found at {image_path}")

# Run YOLO inference
results = model(image_path)

# Copy the image for drawing results
img_with_contours = img.copy()


def smooth_mask(binary_mask, kernel_size=10):
    """ Apply morphological closing to smooth the edges of the binary mask. """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    return cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)


def draw_rounded_contours(binary_mask, canvas, color=(0, 255, 0), thickness=1, simplification_factor=0.001):
    """ Find and simplify contours on the smoothed binary mask. """
    smoothed_mask = smooth_mask(binary_mask, kernel_size=20)
    contours_info = cv2.findContours(smoothed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours_info[0] if len(contours_info) == 2 else contours_info[1]

    rounded_contours = []
    for contour in contours:
        epsilon = simplification_factor * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        rounded_contours.append(approx)

    cv2.drawContours(canvas, rounded_contours, -1, color, thickness)

    # Create a blank mask for the skeletonization step
    mask = np.zeros_like(binary_mask)
    cv2.drawContours(mask, rounded_contours, -1, 255, thickness=cv2.FILLED)

    return mask  # Return mask for further processing


def find_skeleton_endpoints(skeleton):
    """ Identify endpoints as pixels with only one white neighbor. """
    endpoints = []
    h, w = skeleton.shape

    for y in range(1, h - 1):
        for x in range(1, w - 1):
            if skeleton[y, x] == 255:  # White pixel (skeleton)
                # Extract 3x3 neighborhood
                neighbors = skeleton[y - 1:y + 2, x - 1:x + 2]
                count = np.sum(neighbors == 255) - 1  # Exclude center pixel
                if count == 1:  # Only 1 connected white pixel → Endpoint
                    endpoints.append((x, y))

    return endpoints


def find_branch_points(skeleton):
    """ Identify branch points as pixels with 3 or more white neighbors. """
    branch_points = []
    h, w = skeleton.shape

    for y in range(1, h - 1):
        for x in range(1, w - 1):
            if skeleton[y, x] == 255:  # White pixel
                # Extract 3x3 neighborhood
                neighbors = skeleton[y - 1:y + 2, x - 1:x + 2]
                count = np.sum(neighbors == 255) - 1  # Exclude center pixel
                if count >= 3:  # More than 2 connections → Branch point
                    branch_points.append((x, y))

    return branch_points


# Process segmentation masks
if hasattr(results[0], 'masks') and results[0].masks is not None:
    masks = results[0].masks.data  # YOLO segmentation masks
    if hasattr(masks, 'cpu'):
        masks = masks.cpu().numpy()

    # Initialize an empty mask with the same dimensions as the image
    combined_mask = np.zeros(img.shape[:2], dtype=np.uint8)

    # Process each mask and combine them
    for mask in masks:
        mask = np.squeeze(mask)
        binary_mask = (mask > 0.5).astype(np.uint8) * 255
        if binary_mask.shape != img.shape[:2]:
            binary_mask = cv2.resize(binary_mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
        combined_mask = cv2.bitwise_or(combined_mask, binary_mask)

    # Generate mask from the contours
    contour_mask = draw_rounded_contours(combined_mask, img_with_contours)

    # Apply thinning only within the contour mask
    skeleton = np.zeros_like(contour_mask)
    skeleton = xip.thinning(contour_mask, thinningType=0)

    #skeleton = skeletonize(combined_mask)

    # Draw the skeleton on the output image
    skeleton_rgb = cv2.cvtColor(skeleton, cv2.COLOR_GRAY2BGR)  # Convert to RGB
    skeleton_rgb[np.where((skeleton == 255))] = (0, 255, 255)  # Yellow skeleton
    img_with_contours = cv2.addWeighted(img_with_contours, 1, skeleton_rgb, 0.5, 0)

    # Find endpoints and branch points
    end_points = find_skeleton_endpoints(skeleton)
    branch_points = find_branch_points(skeleton)

    # Compute center as the mean of branch points
    if branch_points:
        center = (int(np.mean([p[0] for p in branch_points])), int(np.mean([p[1] for p in branch_points])))
    else:
        center = None

    # Draw endpoints (red)
    #for pt in end_points:
    #    cv2.circle(img_with_contours, pt, 5, (0, 0, 255), -1)

    # Draw center (blue)
    if center:
        cv2.circle(img_with_contours, center, 5, (255, 0, 0), -1)

    # Show final results
    cv2.imshow("Image with Contours, Skeleton, and Endpoints", img_with_contours)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("No segmentation mask detected!")
