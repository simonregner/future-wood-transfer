from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2 import model_zoo

from  mask2former.config import add_maskformer2_config

class Mask2FormerModelLoader:
    def __init__(self):
        """
        Initialize the Mask2Former model.
        Args:
            config_path (str): Path to the config file. If None, uses default COCO instance segmentation config.
            weights_path (str): Path to model weights. If None, uses default pretrained weights.
            device (str): 'cuda' or 'cpu'
        """
        self.cfg = get_cfg()
        add_maskformer2_config(self.cfg)

        self.predictor = None



    def load_model(self, config_path, model_path, device='cuda'):
        """
        Load the Mask2Former model.
        Args:
            config_path (str): Path to the config file.
            weights_path (str): Path to model weights.
            device (str): 'cuda' or 'cpu'
        """
        self.cfg.merge_from_file(config_path)
        self.cfg.merge_from_file(model_zoo.get_config_file("Cityscapes/panoptic_fpn_R_101_dconv.yaml"))


        self.cfg.defrost()

        self.cfg.MODEL.WEIGHTS = model_path
        self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("Cityscapes/panoptic_fpn_R_101_dconv.yaml")
        self.cfg.MODEL.DEVICE = device
        self.cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON = True
        self.cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON = True
        self.cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON = True
        self.cfg.MODEL.MASK_FORMER.TEST.SCORE_THRESH = 0.95

        self.cfg.freeze()

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
        # Get instances
        instances = outputs["instances"]

        # Masks as (N, H, W) tensor of bool
        pred_masks = instances.pred_masks  # shape: [N, H, W], dtype: bool or uint8
        pred_classes = instances.pred_classes  # shape: [N], dtype: int64

        masks_list = []
        classes_list = []

        for mask, cls in zip(pred_masks, pred_classes):
            # Convert mask to numpy array with 0s and 1s
            mask_np = mask.cpu().numpy().astype('uint8')  # Convert bool -> int (0 or 1)
            masks_list.append(mask_np)
            classes_list.append(int(cls.cpu().numpy()))  # Ensure it's a Python int

        print(f"Class IDs: {classes_list}")
        return masks_list, classes_list

# Example usage:
# model = Mask2FormerWrapper()
# result = model.predict(image)