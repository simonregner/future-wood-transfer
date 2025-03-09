from ultralytics import YOLO

# Load a COCO-pretrained YOLO11n model
model = YOLO("runs/segment/yolov12/weights/best.pt")

# Run inference with the YOLO11n model on the 'bus.jpg' image
results = model("test_images/IMG_8566.jpeg", conf=0.5)

for result in results:
    boxes = result.boxes  # Boxes object for bounding box outputs
    masks = result.masks  # Masks object for segmentation masks outputs
    keypoints = result.keypoints  # Keypoints object for pose outputs
    probs = result.probs  # Probs object for classification outputs
    obb = result.obb  # Oriented boxes object for OBB outputs
    result.show()  # display to screen
    result.save(filename="test_results/result_cross_01.jpg")