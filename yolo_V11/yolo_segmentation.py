from ultralytics import YOLO

model = YOLO("yolo12s-seg.yaml")
#model = YOLO("runs/segment/intersection_testing/weights/last.pt")


#result = model.train(data='../data/aifoy1rest_coco8.yaml', epochs=500, batch=3, imgsz=(1280,720), workers=6, device='0')

model.train(data='../data/aiforest_coco8.yaml', batch=18, epochs=500,  workers=12, device='0', dropout=0.3, name="yolov12_flash_attention", cache=True, overlap_mask=False, resume=False, patience=200)