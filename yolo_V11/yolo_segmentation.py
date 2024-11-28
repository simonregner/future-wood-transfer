from ultralytics import YOLO

model = YOLO("yolo11m-seg.pt")

result = model.train(data='../data/aiforest_coco8.yaml', epochs=500, batch=6, imgsz=1024, workers=2)