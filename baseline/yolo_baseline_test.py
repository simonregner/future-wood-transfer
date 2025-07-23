import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from ultralytics import YOLO

# Load YOLOv8 segmentation model
model = YOLO("best.pt")  # Replace with your trained model if needed

# Define allowed classes (YOLO names)
ALLOWED_CLASSES = ["road", "lane"]  # Change to your desired classes
class_name_to_id = {name: i for i, name in enumerate(model.names)}

# Convert class names to IDs
allowed_class_ids = [2,3,4,7]

# List of images
image_paths = [
    "baseline/image_01.png",
    "baseline/image_03.jpg",
    "baseline/image_05.png",
    "baseline/image_06.png",
    "baseline/image_07.png",
    "baseline/image_08.png",    
    "baseline/image_09.jpeg",
    "baseline/image_10.png",
    "baseline/image_11.png",
    "baseline/image_12.png"
]

results = []

for image_path in image_paths:
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width = image_rgb.shape[:2]

    result = model(source=image_rgb, retina_masks=True)[0]  # Single prediction result
    masks = result.masks
    classes = result.boxes.cls.cpu().numpy().astype(int)

    # Create a blank RGB mask image
    seg_mask = np.zeros_like(image_rgb)

    if masks is not None:
        entries = []
        for i, (mask, cls_id) in enumerate(zip(masks.data.cpu().numpy(), classes)):
            if cls_id in allowed_class_ids:
                entries.append((cls_id, mask))

        # Sortieren: zuerst alle außer Klasse 7, dann Klasse 7 zuletzt
        entries = sorted(entries, key=lambda x: x[0] == 7)

        for cls_id, mask in entries:
            binary_mask = cv2.resize((mask > 0.5).astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST)
            color = np.random.randint(0, 255, size=3).tolist()
            for c in range(3):
                seg_mask[:, :, c] = np.where(binary_mask == 1, color[c], seg_mask[:, :, c])

    results.append((image_rgb, seg_mask))

# Plot
rows = len(results)
fig, axs = plt.subplots(rows, 2, figsize=(12, rows * 5))

if rows == 1:
    axs = np.expand_dims(axs, 0)

for i, (orig_img, mask_img) in enumerate(results):
    axs[i, 0].imshow(orig_img)
    axs[i, 0].axis("off")

    axs[i, 1].imshow(mask_img)
    axs[i, 1].axis("off")

plt.tight_layout()
plt.show()