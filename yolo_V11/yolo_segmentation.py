from ultralytics import YOLO

model = YOLO("yolo11m-seg.pt")

result = model.train(data='../data/aiforest_coco8.yaml', epochs=500, batch=3, imgsz=(1280,720), workers=6, device='0')