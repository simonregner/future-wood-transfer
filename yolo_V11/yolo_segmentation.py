from ultralytics import YOLO

if True:
    model = YOLO("yolo12s-seg.yaml")
    #model = YOLO("runs/segment/forest_testarea/weights/last.pt")


    #result = model.train(data='../data/aifoy1rest_coco8.yaml', epochs=500, batch=3, imgsz=(1280,720), workers=6, device='0')

    model.train(data='../data/aiforest_coco8.yaml', batch=0.6, epochs=500, device='0', name="forest_testarea", overlap_mask=False, resume=False, patience=100)
    # rect=true ?
    #model.tune(
    #    data="../data/aiforest_coco8.yaml",
    #    batch=0.7,ä
    #    epochs=20,
    #    iterations=30,
    #    plots=False,
    #    save=False,
    #    val=True,
    #    resume=True,
    #    name="tune",
    #)