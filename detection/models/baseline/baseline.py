from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2 import model_zoo

from  mask2former.config import add_maskformer2_config

from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
import torch

import numpy as np

from scipy.ndimage import label

# Cityscapes label map (label_id to name)
CITYSCAPES_LABELS = {
    0: 'road', 1: 'ego vehicle', 2: 'rectification border', 3: 'out of roi', 4: 'static',
    5: 'dynamic', 6: 'ground', 7: 'road1', 8: 'sidewalk', 9: 'parking', 10: 'rail track',
    11: 'building', 12: 'wall', 13: 'fence', 14: 'guard rail', 15: 'bridge', 16: 'tunnel',
    17: 'pole', 18: 'polegroup', 19: 'traffic light', 20: 'traffic sign', 21: 'vegetation',
    22: 'terrain', 23: 'sky', 24: 'person', 25: 'rider', 26: 'car', 27: 'truck',
    28: 'bus', 29: 'caravan', 30: 'trailer', 31: 'train', 32: 'motorcycle', 33: 'bicycle', -1: 'void'
}



class BaselineModelLoader:
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

        # Load processor and model
        self.processor = AutoImageProcessor.from_pretrained("facebook/mask2former-swin-large-cityscapes-panoptic")
        self.model = Mask2FormerForUniversalSegmentation.from_pretrained("facebook/mask2former-swin-large-cityscapes-panoptic")


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

        # Post-process panoptic segmentation
        inputs = self.processor(images=image, return_tensors="pt")

        # Run inference
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Post-process panoptic segmentation
        result = self.processor.post_process_panoptic_segmentation(outputs, target_sizes=[image.size[::-1]])[0]
        
        # Extract segmentation and segment info
        segmentation_map = result["segmentation"].numpy()
        segments_info = result["segments_info"]

        road_ids = [
            segment["id"]
            for segment in segments_info
            if CITYSCAPES_LABELS.get(segment["label_id"], "") == "road"
        ]

        road_mask = np.isin(segmentation_map, road_ids).astype(np.uint8)


        # Assume road_mask is your binary mask (values 0 or 1)
        labeled_mask, num_features = label(road_mask)

        # Create individual binary masks
        min_area = 1000

        road_masks = []
        road_classes = []

        for i in range(1, num_features + 1):
            component = (labeled_mask == i).astype(np.uint8)
            area = component.sum()
            if area >= min_area:
                road_masks.append(component)
                road_classes.append(4)


        
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