import numpy as np
from ultralytics import YOLO


class YOLODetectionLoader:
    """
    Loads and runs a YOLO detection model.
    Returns bounding boxes, class IDs, confidences, and class names for each frame.
    """

    model = None
    names = None
    imgsz = 640
    half  = False

    @classmethod
    def load_model(cls, model_path="YOLO26s-seg.pt", imgsz=640, half=False):
        cls.model = YOLO(model_path)
        cls.names = cls.model.names
        cls.imgsz = imgsz
        cls.half  = half

        # Warmup: triggers CUDA JIT / TensorRT init so the first real frame is fast.
        dummy = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)
        cls.model.predict(source=dummy, imgsz=imgsz, half=half,
                          retina_masks=True, verbose=False)

    @classmethod
    def predict(cls, image, conf=0.5, classes=None):
        """
        Run detection/segmentation on an image.

        Args:
            image: BGR numpy array.
            conf: Minimum confidence threshold.
            classes: List of class IDs to filter for (e.g. [0, 2] for person+car).
                     None means all classes.

        Returns:
            boxes_xyxy: (N, 4) float array of [x1, y1, x2, y2] pixel coordinates.
            class_ids: (N,) int array of class IDs.
            confidences: (N,) float array of confidence scores.
            class_names: list of N class name strings.
            masks: (N, H, W) uint8 array of binary masks (0/255), or None if the
                   model is a detection-only variant without segmentation output.
        """
        results = cls.model.predict(source=image, conf=conf, classes=classes,
                                    imgsz=cls.imgsz, half=cls.half,
                                    retina_masks=True, verbose=False)

        result = results[0]
        if result.boxes is None or len(result.boxes) == 0:
            return np.empty((0, 4)), np.empty(0, dtype=int), np.empty(0), [], None

        boxes_xyxy = result.boxes.xyxy.cpu().numpy()           # (N, 4)
        class_ids  = result.boxes.cls.cpu().numpy().astype(int) # (N,)
        confidences = result.boxes.conf.cpu().numpy()           # (N,)
        class_names = [cls.names[c] for c in class_ids]

        if result.masks is not None:
            masks = result.masks.data.cpu().numpy()             # (N, H, W) float 0-1
            masks = (masks * 255).astype(np.uint8)
        else:
            masks = None

        return boxes_xyxy, class_ids, confidences, class_names, masks
