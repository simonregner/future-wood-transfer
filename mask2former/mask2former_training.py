import os
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer, default_setup, launch
from train_net import Trainer
from detectron2.data.datasets import register_coco_panoptic_separated, register_coco_panoptic, register_coco_instances
from detectron2.utils.logger import setup_logger

from mask2former.config import add_maskformer2_config
from detectron2.model_zoo import get_config_file

def register_panoptic_datasets(dataset_root):
    # Register panoptic training/validation datasets
    if False:
        register_coco_instances(
            "fwt_instance_train",
            metadata={"thing_classes": ['background',
                                        'bugle_road', 'left_turn', 'right_turn', 'road', 'straight_turn', 'intersection', 'lane']},
            json_file=os.path.join(dataset_root, "annotations/instances_train.json"),
            image_root=os.path.join(dataset_root, "train")
        )
        register_coco_instances(
            "fwt_instance_val",
            metadata={"thing_classes": ['background','bugle_road', 'left_turn', 'right_turn', 'road', 'straight_turn', 'intersection', 'lane']},
            json_file=os.path.join(dataset_root, "annotations/instances_val.json"),
            image_root=os.path.join(dataset_root, "val")
        )
    if True:
        register_coco_instances(
            "fwt_instance_train",
            metadata={"thing_classes": ['road', 'road_boundary']},
            json_file=os.path.join(dataset_root, "annotations/instances_train.json"),
            image_root=os.path.join(dataset_root, "train")
        )
        register_coco_instances(
            "fwt_instance_val",
            metadata={"thing_classes": ['road', 'road_boundary']},
            json_file=os.path.join(dataset_root, "annotations/instances_val.json"),
            image_root=os.path.join(dataset_root, "val")
        )

    '''register_coco_panoptic(
        name="fwt_panoptic_train",
        metadata={"thing_classes":["class_1", "class_2", "class_3", "class_4", "class_5", "class_6", "class_7", "class_8"]},
        image_root=os.path.join(dataset_root, "train"),
        #panoptic_root=os.path.join(dataset_root, "panoptic_train"),
        panoptic_json=os.path.join(dataset_root, "results.json"),
        #sem_seg_root=None,
        instances_json=None
    )

    #register_coco_panoptic(
    #    name="fwt_panoptic_val",
    ##    metadata={"thing_classes":["class_1", "class_2", "class_3", "class_4", "class_5", "class_6", "class_7", "class_8"]},
    #    image_root=os.path.join(dataset_root, "val"),
    #    panoptic_root=os.path.join(dataset_root, "panoptic_val"),
    #    panoptic_json=os.path.join(dataset_root, "annotations/panoptic_val.json"),
    #    #sem_seg_root=None,
    #    instances_json=None
    #)
    '''


def setup_cfg(output_dir):
    cfg = get_cfg()
    cfg.set_new_allowed(True)
    add_maskformer2_config(cfg)

    cfg.merge_from_file("/home/simon/Documents/Master-Thesis/future-wood-transfer/mask2former/Mask2Former/configs/coco/instance-segmentation/maskformer2_R50_bs16_50ep.yaml")
    # Remove any reference to MODEL.RESNETS.STEM_TYPE if present in your config file,
    # or ensure you are using a config file compatible with Swin Transformer backbone.

    cfg.DATASETS.TRAIN = ("fwt_instance_train",)
    cfg.DATASETS.TEST = ("fwt_instance_val",)
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.001
    cfg.SOLVER.MAX_ITER = 5000
    cfg.SOLVER.STEPS = []
    cfg.SOLVER.CHECKPOINT_PERIOD = 5000

    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2  # change this!
    cfg.MODEL.DROPOUT = 0.5

    cfg.TEST.DETECTIONS_PER_IMAGE = 50

    cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON = False
    cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON = True  # only if you need instance masks
    cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON = False

    cfg.SOLVER.AMP.ENABLED = True

        # Number of classes for instance segmentation head
    # (Mask2Former uses SEM_SEG_HEAD.NUM_CLASSES for instance/panoptic too)
    cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = 2

    # ROI settings don't apply here; Mask2Former is query-based.
    # But some configs use this value in evaluation code paths; keep consistent:
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2
    cfg.MODEL.WEIGHTS = ""  # <- key line (random init)


    cfg.INPUT.MASK_FORMAT = "bitmask"

     # Limit training image size to save VRAM
    cfg.INPUT.MIN_SIZE_TRAIN = (256, 384, 512)   # randomly picks one → helps generalization
    cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING = "choice"
    cfg.INPUT.MAX_SIZE_TRAIN = 768

    cfg.INPUT.MIN_SIZE_TEST = 512
    cfg.INPUT.MAX_SIZE_TEST = 768

    cfg.OUTPUT_DIR = output_dir
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    return cfg


def main():
    dataset_root = "/home/simon/Documents/Master-Thesis/data/coco_training_data_road"
    output_dir = "/home/simon/Documents/Master-Thesis/future-wood-transfer/mask2former/output/paper_road_test/"

    register_panoptic_datasets(dataset_root)

    cfg = setup_cfg(output_dir)
    setup_logger(output=output_dir)
    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=True)
    trainer.train()

if __name__ == "__main__":
    launch(
        main,
        num_gpus_per_machine=1,
        num_machines=1,
        dist_url="auto"
    )