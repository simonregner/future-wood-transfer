from ultralytics import YOLO

model = YOLO("yolo11m-seg.pt")

#result = model.train(data='../data/aiforest_coco8.yaml', epochs=500, batch=3, imgsz=(1280,720), workers=6, device='0')

model.tune(data='../data/aiforest_coco8.yaml', batch=12, epochs=50, iterations=100, workers=4)