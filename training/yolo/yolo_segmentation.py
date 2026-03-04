from ultralytics import YOLO#

if True:
    model = YOLO("yolo11m-seg.pt")
    #model = YOLO("yolo12s-seg.yaml")
    #model = YOLO("runs/segment/model_test_v11_m/weights/last.pt")


    #result = model.train(data='../data/aifoy1rest_coco8.yaml', epochs=500, batch=3, imgsz=(1280,720), workers=6, device='0')

    model.train(data='../data/aiforest_road.yaml',
                batch=0.7,
                epochs=500,
                device='0',
                name="paper_road_big_line_2",
                overlap_mask=False,
                resume=False,
                patience=10,
                dropout=0.5,
                multi_scale=True,
                optimizer="auto",
                #cfg="best_hyperparameters.yaml",
                freeze=5,  # Freeze first 5 layers
                nbs=64, 
                warmup_epochs=10,
                #imgsz=320,
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