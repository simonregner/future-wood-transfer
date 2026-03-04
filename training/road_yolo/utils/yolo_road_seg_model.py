import torch.nn as nn
from ultralytics.nn.tasks import DetectionModel

# -----------------------------
# YOLOv12 Road Segmentation Model
# -----------------------------
class YOLO12RoadSegModel(DetectionModel):
    """
    Extends the YOLOv12 segmentation model with an additional road-grouping head.
    """
    def __init__(self, cfg='yolo12s-seg.yaml', nc=8, num_road_groups=10, verbose=True):
        super().__init__(cfg=cfg, nc=nc, verbose=verbose)
        self.num_road_groups = num_road_groups

        # Add road group prediction head
        self.road_group_head = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, num_road_groups, kernel_size=1)
        )

    def forward(self, x, *args, **kwargs):
        if isinstance(x, dict):
            # Training: x is a dict containing 'img' and other keys
            x_in = x['img'] if 'img' in x else x['image']
            x = self.model(x_in)  # backbone + head
            self.seg = self.model[-1]

            # Segmentation loss
            loss = self.loss(x, *args, **kwargs)

            # Road group output from high-res feature map
            feature_map = x[14]  # P3 layer
            road_group_out = self.road_group_head(feature_map)

            return loss, road_group_out

        else:
            # Inference
            return self.predict(x, *args, **kwargs)
