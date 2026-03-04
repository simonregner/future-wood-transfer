import cv2
import numpy as np
from ultralytics import YOLO  # YOLO model

# Set image and model paths
image_path = "2022-10-12_solalinden_waldwege__0095_1665579514737884132_windshield_vis.png"
model = YOLO("../yolo_V11/runs/segment/intersection_testing/weights/best.pt")
#model = YOLO("../yolo_V11/runs/segment/train4/weights/best.pt")


# Load the image
img = cv2.imread(image_path)
if img is None:
    raise ValueError(f"Image not found at {image_path}")

# Run YOLO inference
results = model.predict(img, agnostic_nms=True, retina_masks=True, max_det=10)

# -------------------------------
# Approach 1: Using the built-in rendering
# -------------------------------
# The .plot() method returns the image with predictions drawn.
img_rendered = results[0].plot()  # Assumes at least one result is returned

# Display the rendered image
cv2.imshow("YOLO Inference (Built-in Render)", img_rendered)


# -------------------------------
# Approach 2: Manually overlaying segmentation masks (if available)
# -------------------------------
# Check if the result contains segmentation masks.
# Note: This assumes that your model outputs segmentation masks in results[0].masks.
if hasattr(results[0], 'masks') and results[0].masks is not None:
    # Typically, results[0].masks.data is a tensor or array of masks.
    # We'll use the first mask for demonstration. Adjust if there are multiple masks.
    mask = results[0].masks.data[0].cpu().numpy() if hasattr(results[0].masks.data[0], 'cpu') else \
    results[0].masks.data[0]

    # Normalize mask values to the range [0, 255] and convert to uint8.
    # This assumes that the mask values are in the range [0, 1]. Adjust if needed.
    mask = (mask * 255).astype(np.uint8)

    # Ensure mask has the same spatial dimensions as the image.
    # If necessary, resize the mask to match the image dimensions.
    if mask.shape[:2] != img.shape[:2]:
        mask = cv2.resize(mask, (img.shape[1], img.shape[0]))

    # Apply a color map to the mask for better visualization.
    colored_mask = cv2.applyColorMap(mask, cv2.COLORMAP_JET)

    # Blend the original image and the colored mask.
    alpha = 0.5  # Adjust transparency: 0.0 (only original) to 1.0 (only mask)
    overlay = cv2.addWeighted(img, 1 - alpha, colored_mask, alpha, 0)

    # Display the overlay image.
    cv2.imshow("YOLO Inference (Mask Overlay)", overlay)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("No segmentation masks available in the result. Only bounding boxes might have been detected.")
