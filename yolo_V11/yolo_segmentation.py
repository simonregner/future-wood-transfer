from ultralytics import YOLO#

if True:
    model = YOLO("yolo11s-seg.pt")
    #model = YOLO("runs/segment/MM_Dataset/weights/best.pt")


    #result = model.train(data='../data/aifoy1rest_coco8.yaml', epochs=500, batch=3, imgsz=(1280,720), workers=6, device='0')

    model.train(data='../data/aiforest_coco8.yaml',
                batch=0.7,
                epochs=400,
                device='0',
                name="MM_auto_labeling",
                overlap_mask=False,
                resume=False,
                patience=100,
                dropout=0.4,
                multi_scale=True
                )
    # rect=true ?
    #model.tune(
    #    data="../data/aiforest_coco8.yaml",
    #    batch=0.7,
    #    epochs=20,
    #    iterations=30,
    #    plots=False,
    #    save=False,
    #    val=True,
    #    resume=True,
    #    name="tune",
    #)