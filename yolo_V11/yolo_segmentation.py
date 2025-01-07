from ultralytics import YOLO

model = YOLO("yolo11s-seg.pt")
#model = YOLO("/home/simon/Documents/Master-Thesis/mt-aiforest/yolo_V11/runs/segment/train3/weights/last.pt")


#result = model.train(data='../data/aiforest_coco8.yaml', epochs=500, batch=3, imgsz=(1280,720), workers=6, device='0')

model.train(data='../data/aiforest_coco8.yaml', batch=30, epochs=500,  workers=12, device='0', dropout=0.3)