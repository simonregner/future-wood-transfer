from ultralytics import YOLO
import numpy as np


class YOLOModelLoader:
    """
    A global class to manage the loading and inference of a YOLOv11 model.
    """
    model = None  # Class variable to store the YOLO model instance

    @classmethod
    def load_model(cls, model_path="best.pt"):
        """
        Load the YOLOv11 model if not already loaded.

        Args:
            model_path (str): Path to the YOLO model file.
        """
        if cls.model is None:
            print(f"Loading YOLO model from {model_path}...")
            cls.model = YOLO(model_path)
            print("Model loaded successfully.")
        else:
            print("Model is already loaded.")

    @classmethod
    def predict(cls, image, conf=0.70):
        """
        Perform inference using the loaded YOLO model.

        Args:
            image (str): Path to the image to run inference on.
            conf (float): Confidence threshold for predictions.

        Returns:
            results: masks, classes.
        """
        if cls.model is None:
            raise ValueError("Model is not loaded. Please call load_model() first.")
        if type(image) is not str:
            image = np.array(image, copy=False, dtype=np.uint8)

        print(f"Running inference on image with shape: {image.shape} and dtype: {image.dtype}")
        image = np.asarray(image).copy()
        assert isinstance(image, np.ndarray)
        results = cls.model.predict(source=image, conf=conf, retina_masks=True)#,  classes=[7])#, agnostic_nms=True, retina_masks=True)

        if len(results[0].boxes) == 0:
            return [], []

        masks = results[0].masks.data.cpu().numpy().astype(np.uint8) * 255
        classes = results[0].boxes.cls.cpu().numpy().astype(np.uint8)
        return masks, classes
