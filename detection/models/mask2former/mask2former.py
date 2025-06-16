import torch
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor

class Mask2FormerWrapper:
    def __init__(self, config_path=None, weights_path=None, device='cuda'):
        """
        Initialize the Mask2Former model.
        Args:
            config_path (str): Path to the config file. If None, uses default COCO instance segmentation config.
            weights_path (str): Path to model weights. If None, uses default pretrained weights.
            device (str): 'cuda' or 'cpu'
        """
        self.cfg = get_cfg()
        # Use default config if none provided
        if config_path is None:
            from detectron2.model_zoo import get_config_file, get_checkpoint_url
            config_path = "COCO-InstanceSegmentation/mask2former_R50_bs16_50ep.yaml"
            self.cfg.merge_from_file(get_config_file(config_path))
            weights_path = get_checkpoint_url(config_path)
        else:
            self.cfg.merge_from_file(config_path)
        if weights_path is not None:
            self.cfg.MODEL.WEIGHTS = weights_path
        self.cfg.MODEL.DEVICE = device
        self.predictor = DefaultPredictor(self.cfg)

    def predict(self, image):
        """
        Run inference on an image.
        Args:
            image (np.ndarray): Input image in BGR format (as read by OpenCV).
        Returns:
            dict: Prediction results.
        """
        outputs = self.predictor(image)
        return outputs

# Example usage:
# model = Mask2FormerWrapper()
# result = model.predict(image)