from ultralytics import YOLO#

model = YOLO("yolo11m-seg.pt")

# rect=true ?

search_space = {
    "lr0": (1e-5, 1e-1),
    "lrf": (0.01, 1.0),
    "momentum": (0.6, 0.999),
    "weight_decay": (0.0, 0.001),
    "warmup_epochs": (0.0, 5.0),
    "warmup_momentum": (0.0, 0.95),
    "box": (0.02, 0.2),
    "cls": (0.2, 4.0),
    #"hsv_h": (0.0, 0.1),
    #"hsv_s": (0.0, 0.9),
    #"hsv_v": (0.0, 0.9),
    "degrees": (45.0, 180.0),
    "translate": (0.1, 0.9),
    "scale": (0.1, 0.9),
    "shear": (1.0, 10.0),
    "perspective": (0.0, 0.001),
    #"flipud": (0.0, 1.0),
    #"fliplr": (0.0, 1.0),
    "mosaic": (0.1, 1.0),
    "mixup": (0.1, 1.0),
    "copy_paste": (0.1, 1.0)
}

model.tune(
    data="../data/aiforest_road_noaug.yaml",
    batch=0.6,
    epochs=50,
    iterations=500,
    imgsz=320,
    half=True,
    plots=False,
    save=False,
    resume=True,
    space=search_space,
    name="paper_tune",
    fraction=0.5,
    warmup_epochs=3,
)