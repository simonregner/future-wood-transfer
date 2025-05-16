import os
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer, default_setup, launch
from detectron2.data.datasets import register_coco_panoptic_separated
from detectron2.utils.logger import setup_logger

def register_panoptic_datasets(dataset_root):
    # Register panoptic training/validation datasets
    register_coco_panoptic_separated(
        name="my_panoptic_train",
        metadata={},
        image_root=os.path.join(dataset_root, "train2017"),
        panoptic_root=os.path.join(dataset_root, "panoptic_train2017"),
        panoptic_json=os.path.join(dataset_root, "annotations/panoptic_train2017.json"),
        sem_seg_root=None,
        instances_json=os.path.join(dataset_root, "annotations/instances_train2017.json")
    )

    register_coco_panoptic_separated(
        name="my_panoptic_val",
        metadata={},
        image_root=os.path.join(dataset_root, "val2017"),
        panoptic_root=os.path.join(dataset_root, "panoptic_val2017"),
        panoptic_json=os.path.join(dataset_root, "annotations/panoptic_val2017.json"),
        sem_seg_root=None,
        instances_json=os.path.join(dataset_root, "annotations/instances_val2017.json")
    )


def setup_cfg(output_dir):
    cfg = get_cfg()
    from mask2former import add_maskformer2_config
    add_maskformer2_config(cfg)

    cfg.merge_from_file("configs/coco/panoptic-segmentation/swin/maskformer2_swin_base_IN21k_384_bs16_50ep.yaml")

    cfg.DATASETS.TRAIN = ("my_panoptic_train",)
    cfg.DATASETS.TEST = ("my_panoptic_val",)
    cfg.DATALOADER.NUM_WORKERS = 4
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.0001
    cfg.SOLVER.MAX_ITER = 10000
    cfg.SOLVER.STEPS = []
    cfg.SOLVER.CHECKPOINT_PERIOD = 1000

    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 8  # change this!
    cfg.INPUT.MASK_FORMAT = "bitmask"

    cfg.OUTPUT_DIR = output_dir
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    return cfg


def main():
    dataset_root = "/home/simon/Documents/Master-Thesis/data/yolo_training_data"
    output_dir = "./output_mask2former_panoptic"

    register_panoptic_datasets(dataset_root)

    cfg = setup_cfg(output_dir)
    setup_logger(output=output_dir)
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()

if __name__ == "__main__":
    launch(
        main,
        num_gpus_per_machine=1,
        num_machines=1,
        dist_url="auto"
    )