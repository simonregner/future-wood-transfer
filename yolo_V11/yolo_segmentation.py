from ultralytics import YOLO

model = YOLO("yolo11m-seg.pt")

result = model.train(data='../data/aiforest_coco8.yaml', epochs=200, batch=8)