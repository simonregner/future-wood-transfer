from ultralytics import YOLO#

if True:
    model = YOLO("yolo11s-seg.pt")
    #model = YOLO("yolo12s-seg.yaml")
    #model = YOLO("runs/segment/MM_freeze/weights/last.pt")


    #result = model.train(data='../data/aifoy1rest_coco8.yaml', epochs=500, batch=3, imgsz=(1280,720), workers=6, device='0')

    model.train(data='../data/aiforest_coco8.yaml',
                batch=0.7,
                epochs=100,
                device='0',
                name="model_test_v11_d0.2_freeze01",
                overlap_mask=False,
                resume=False,
                patience=100,
                dropout=0.5,
                multi_scale=True,
                optimizer="auto",
                cfg="best_hyperparameters.yaml",
                freeze=[0, 1, 2, 3, 4],  # Freeze first 5 layers
                nbs=64
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