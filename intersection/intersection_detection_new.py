import cv2
import numpy as np
from ultralytics import YOLO  # YOLO model
import cv2.ximgproc as xip  # For skeleton thinning
from skimage.morphology import skeletonize

# Set image and model paths
image_path = "2022-10-12_solalinden_waldwege__0095_1665579514737884132_windshield_vis.png"
model = YOLO("../detection/best.pt")

# Load the image
img = cv2.imread(image_path)
if img is None:
    raise ValueError(f"Image not found at {image_path}")

# Run YOLO inference
results = model(image_path, retina_masks=True)

# Copy the image for drawing results
img_with_contours = img.copy()

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

    skeleton = skeletonize(combined_mask)

    # Draw the skeleton on the output image
    skeleton_rgb = cv2.cvtColor(skeleton, cv2.COLOR_GRAY2BGR)  # Convert to RGB
    skeleton_rgb[np.where((skeleton == 255))] = (0, 255, 255)  # Yellow skeleton
    img_with_contours = cv2.addWeighted(img_with_contours, 1, skeleton_rgb, 0.5, 0)

    # Show final results
    cv2.imshow("Image with Contours, Skeleton, and Endpoints", img_with_contours)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

