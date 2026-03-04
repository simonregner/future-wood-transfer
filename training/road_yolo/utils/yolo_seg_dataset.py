import os

import torch
from torch.utils.data import Dataset

import cv2
import numpy as np



# -----------------------------
# Dataset for YOLO polygon segmentation with road index
# -----------------------------
class YoloTxtSegDataset(Dataset):
    def __init__(self, txt_dir, image_dir, image_size=(640, 640), include_classes=None):
        self.txt_dir = txt_dir
        self.image_dir = image_dir
        self.image_size = image_size
        self.include_classes = include_classes if include_classes is not None else []
        self.samples = [f for f in os.listdir(txt_dir) if f.endswith(".txt")]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        txt_file = os.path.join(self.txt_dir, self.samples[idx])
        img_name_base = self.samples[idx].replace(".txt", "")

        img_file = None
        for ext in [".jpg", ".png", ".jpeg", ".bmp", ".tif", ".tiff"]:
            candidate = os.path.join(self.image_dir, img_name_base + ext)
            if os.path.isfile(candidate):
                img_file = candidate
                break
        if img_file is None:
            raise FileNotFoundError(f"Image file not found for base name: {img_name_base}")

        img = cv2.imread(img_file)
        if img is None:
            raise FileNotFoundError(f"Image file not found or could not be read: {img_file}")
        img = cv2.resize(img, self.image_size)
        img = img.transpose(2, 0, 1) / 255.0
        img = torch.tensor(img, dtype=torch.float32)

        masks = []
        class_ids = []
        road_ids = []

        with open(txt_file, 'r') as f:
            for line in f:
                tokens = line.strip().split()
                if len(tokens) < 6 or len(tokens) % 2 != 0:
                    continue
                class_id = int(tokens[0])
                if self.include_classes and class_id not in self.include_classes:
                    continue
                road_id = int(tokens[1])
                coords = list(map(float, tokens[2:]))
                polygon = np.array(coords).reshape(-1, 2) * np.array(self.image_size[::-1])
                mask = np.zeros(self.image_size[::-1], dtype=np.uint8)
                pts = np.round(polygon).astype(np.int32)
                cv2.fillPoly(mask, [pts], 1)
                masks.append(mask)
                class_ids.append(class_id)
                road_ids.append(road_id)

        if masks:
            masks = torch.tensor(np.stack(masks), dtype=torch.float32)
        else:
            masks = torch.zeros((1, *self.image_size[::-1]), dtype=torch.float32)
        class_ids = torch.tensor(class_ids, dtype=torch.long)
        road_ids = torch.tensor(road_ids, dtype=torch.long)

        return img, masks, class_ids, road_ids