import os
import cv2
import shutil
import numpy as np
from scipy.interpolate import splprep, splev
from ultralytics import YOLO

save_mode = "original"

# --- CONFIGURATION ---
image_folder = "../../SVO/MM_ForestRoads_02_2/old_images"  # 👈 Set this to your folder
confidence_threshold = 0.9
desired_class_ids = [2, 3, 4, 5]  # 👈 Set class IDs you care about

# --- RELATIVE PATH SETUP ---
label_folder = os.path.abspath(os.path.join(image_folder, "../labels"))
filtered_image_folder = os.path.join(image_folder, "../images")

os.makedirs(label_folder, exist_ok=True)
os.makedirs(filtered_image_folder, exist_ok=True)

# Load YOLO segmentation model
model = YOLO("runs/segment/MM_Dataset/weights/best.pt")  # 👈 Change to your actual model path

def smooth_polygon(poly, smoothing_factor=0.00001):
    if len(poly) < 4:
        return poly
    x, y = poly[:, 0], poly[:, 1]
    try:
        tck, u = splprep([x, y], s=0.00001, per=True)
        u_new = np.linspace(0, 1, 200)
        x_new, y_new = splev(u_new, tck)
        return np.stack([x_new, y_new], axis=1)
    except Exception as e:
        print("Smoothing failed:", e)
        return poly

# Process each image
for img_name in os.listdir(image_folder):
    if not img_name.lower().endswith((".png", ".jpg", ".jpeg")):
        continue

    img_path = os.path.join(image_folder, img_name)
    results = model.predict(img_path, classes=desired_class_ids, conf=confidence_threshold, retina_masks=True)[0]

    h, w = results.orig_shape[:2]
    label_lines = []

    # 🛡️ Check if there are any masks or boxes before continuing
    if results.masks is None or results.boxes is None:
        print(f"✘ {img_name} → no detections found")
        continue

    for mask, cls_id in zip(results.masks.data, results.boxes.cls):
        cls_id = int(cls_id.item())

        mask_np = mask.cpu().numpy().astype(np.uint8) * 255

        mh, mw = mask_np.shape  # use mask's own resolution for normalization

        contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:2]

        for contour in contours:
            if len(contour) < 3:
                continue
            contour = contour.squeeze()
            if contour.ndim != 2:
                continue

            flat = " ".join([f"{x / mw:.6f} {y / mh:.6f}" for x, y in contour])
            label_lines.append(f"{cls_id} {flat}")

    if label_lines:
        # Save label file (with same name as image, .txt)
        label_path = os.path.join(label_folder, os.path.splitext(img_name)[0] + ".txt")

        if os.path.exists(label_path):
            print(f"⏩ {img_name} → skipped (label already exists)")
            continue

        with open(label_path, "w") as f:
            f.write("\n".join(label_lines))

        # Save based on configuration
        if save_mode in ("original", "both"):
            shutil.copy(img_path, os.path.join(filtered_image_folder, img_name))

        if save_mode in ("prediction", "both"):
            # YOLO will save to a subfolder in current working dir by default, so we specify full path
            result_img_name = f"pred_{img_name}" if save_mode == "both" else img_name
            result_save_path = os.path.join(filtered_image_folder, result_img_name)
            results.save(filename=result_save_path)

        print(f"✔ {img_name} → saved ({save_mode})")
    else:
        print(f"✘ {img_name} → no desired classes detected")
