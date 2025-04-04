from ultralytics import YOLO

if True:
    model = YOLO("yolo12s-seg.yaml")
    # = YOLO("runs/segment/lane_detection_test/weights/last.pt")


    #result = model.train(data='../data/aifoy1rest_coco8.yaml', epochs=500, batch=3, imgsz=(1280,720), workers=6, device='0')

    model.train(data='../data/aiforest_coco8.yaml', batch=0.7, epochs=400,  workers=10, device='0', dropout=0.3, name="only_lane_detection", cache=True, overlap_mask=False, resume=False, patience=200, classes=[7])