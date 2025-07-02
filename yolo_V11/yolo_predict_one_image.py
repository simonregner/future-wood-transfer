from ultralytics import YOLO
import cv2
import numpy as np


# Load a model
model = YOLO("yolo_V11/runs/segment/forest_testarea/weights/best.pt")  # pretrained YOLO11n model

# Run batched inference on a list of images
results = model(["yolo_V11/test_images/001.jpg"])  # return a list of Results objects

# Process results list
for result in results:
    #boxes = result.boxes  # Boxes object for bounding box outputs
    masks = result.masks  # Masks object for segmentation masks outputs
    keypoints = result.keypoints  # Keypoints object for pose outputs
    probs = result.probs  # Probs object for classification outputs
    obb = result.obb  # Oriented boxes object for OBB outputs
    result.show()  # display to screen
    result.save(filename="result.jpg")  # save to disk