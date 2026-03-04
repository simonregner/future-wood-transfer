# train_mask2former_coco_from_scratch.py
import os
import json
import argparse
from typing import Optional

import torch
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultTrainer, default_setup, launch
from detectron2.evaluation import COCOEvaluator
from detectron2.data.datasets import register_coco_instances

# Mask2Former config hook
from mask2former.config import add_maskformer2_config

from detectron2.data import build_detection_train_loader, build_detection_test_loader
from mask2former.data.dataset_mappers.mask_former_instance_dataset_mapper import (
    MaskFormerInstanceDatasetMapper,
)


class Mask2FormerInstanceTrainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name, cfg, True, output_folder)

    @classmethod
    def build_train_loader(cls, cfg):
        mapper = MaskFormerInstanceDatasetMapper(
            cfg,
            is_train=True,
            # You can override defaults here if needed:
            # augmentations are read from cfg.INPUT / cfg.INPUT.CROP, etc.
        )
        return build_detection_train_loader(cfg, mapper=mapper)

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        # test-time mapper is created inside build_detection_test_loader
        return build_detection_test_loader(cfg, dataset_name)


def register_datasets(train_name: str, val_name: str, train_json: str, train_img: str, val_json: str, val_img: str):
    assert os.path.isfile(train_json), f"train json not found: {train_json}"
    assert os.path.isdir(train_img), f"train images dir not found: {train_img}"
    assert os.path.isfile(val_json), f"val json not found: {val_json}"
    assert os.path.isdir(val_img), f"val images dir not found: {val_img}"

    register_coco_instances(train_name, {}, train_json, train_img)
    register_coco_instances(val_name, {}, val_json, val_img)


def infer_num_classes_from_json(coco_json_path: str) -> int:
    with open(coco_json_path, "r") as f:
        data = json.load(f)
    cats = data.get("categories", [])
    # Using distinct category ids to be robust
    return len({c["id"] for c in cats})


def setup_cfg(
    output_dir: str,
    base_config: Optional[str],
    train_name: str,
    val_name: str,
    num_classes: int,
    ims_per_batch: int,
    base_lr: float,
    max_iter: int,
    warmup_iters: int,
    num_workers: int,
    image_size_train: int,
    image_size_test: int,
    batch_size_per_image: int,
    num_queries: int,
    seed: int,
):
    """
    Build a config to train Mask2Former instance seg FROM SCRATCH (no pretrained weights).
    """
    cfg = get_cfg()
    cfg.set_new_allowed(True)
    add_maskformer2_config(cfg)

    # If you already have the Mask2Former repo, this is a good default instance-seg config:
    # configs path example (adjust to your checkout):
    default_base = "./Mask2Former/configs/coco/instance-segmentation/maskformer2_R50_bs16_50ep.yaml"
    cfg_path = base_config if base_config is not None else default_base
    cfg.merge_from_file(cfg_path)

    # Datasets
    cfg.DATASETS.TRAIN = (train_name,)
    cfg.DATASETS.TEST = (val_name,)

    # Input (COCO polygons)
    cfg.INPUT.MASK_FORMAT = "polygon"
    cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS = True
    cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON = False
    cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON = True
    cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON = False
    # optional: set shorter resize edge for speed/memory; keep aspect ratio
    cfg.INPUT.MIN_SIZE_TRAIN = (image_size_train,)
    cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING = "choice"
    cfg.INPUT.MIN_SIZE_TEST = image_size_test
    cfg.INPUT.MAX_SIZE_TRAIN = max(1333, image_size_train)
    cfg.INPUT.MAX_SIZE_TEST = max(1333, image_size_test)

    # Dataloader
    cfg.DATALOADER.NUM_WORKERS = num_workers

    # Solver / schedule
    cfg.SOLVER.IMS_PER_BATCH = ims_per_batch
    cfg.SOLVER.BASE_LR = base_lr
    cfg.SOLVER.MAX_ITER = max_iter
    cfg.SOLVER.WARMUP_ITERS = warmup_iters
    cfg.SOLVER.STEPS = []  # no LR decay by default; you can set e.g. [60000, 80000]
    cfg.SOLVER.CHECKPOINT_PERIOD = max(1000, max_iter // 10)

    # Random init: ensure we do NOT load pretrained weights
    cfg.MODEL.WEIGHTS = ""  # <- key line (random init)
    # do not freeze backbone when training from scratch
    cfg.MODEL.BACKBONE.FREEZE_AT = 0

    # Mask2Former-specific knobs
    cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES = num_queries
    # Number of classes for instance segmentation head
    # (Mask2Former uses SEM_SEG_HEAD.NUM_CLASSES for instance/panoptic too)
    cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = num_classes

    # ROI settings don't apply here; Mask2Former is query-based.
    # But some configs use this value in evaluation code paths; keep consistent:
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes

    # Train-time batch samples
    cfg.SOLVER.BATCH_SIZE_PER_IMAGE = batch_size_per_image  # used by some samplers (not critical here)

    # Test
    cfg.TEST.DETECTIONS_PER_IMAGE = 32

    cfg.SOLVER.CLIP_GRADIENTS.ENABLED = True
    cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE = "value"   # or "norm" if your build supports it
    cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE = 1.0      # only used when CLIP_TYPE == "value"
    cfg.SOLVER.CLIP_GRADIENTS.NORM_TYPE  = 2.0      # only used when CLIP_TYPE == "norm"

    # Output/seed
    cfg.OUTPUT_DIR = output_dir
    cfg.SEED = seed

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    return cfg


def parse_args():
    p = argparse.ArgumentParser("Train Mask2Former (Instance Seg) from scratch on COCO-style data")
    p.add_argument("--train-json", default="/home/simon/Documents/Master-Thesis/data/coco_training_data_road/annotations/instances_train.json", help="Path to COCO train annotations json")
    p.add_argument("--train-img", default="/home/simon/Documents/Master-Thesis/data/coco_training_data_road/train/", help="Folder with train images")
    p.add_argument("--val-json", default="/home/simon/Documents/Master-Thesis/data/coco_training_data_road/annotations/instances_val.json", help="Path to COCO val annotations json")
    p.add_argument("--val-img", default="/home/simon/Documents/Master-Thesis/data/coco_training_data_road/val/", help="Folder with val images")
    p.add_argument("--output", default="./output/paper_road_roadboundary", help="Output directory")
    p.add_argument("--config", default=None, help="Path to a Mask2Former instance-seg yaml (optional)")
    p.add_argument("--ims-per-batch", type=int, default=2, help="Global batch size (sum over GPUs)")
    p.add_argument("--base-lr", type=float, default=0.0005, help="Base learning rate")
    p.add_argument("--max-iter", type=int, default=5000, help="Training iterations")
    p.add_argument("--warmup-iters", type=int, default=1000, help="Warmup iterations")
    p.add_argument("--num-workers", type=int, default=2, help="Data loader workers")
    p.add_argument("--image-size-train", type=int, default=512, help="Short edge train resize")
    p.add_argument("--image-size-test", type=int, default=512, help="Short edge test resize")
    p.add_argument("--batch-size-per-image", type=int, default=32, help="Samples per image (not critical for M2F)")
    p.add_argument("--num-queries", type=int, default=100, help="Mask2Former object queries")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--num-gpus", type=int, default=1, help="Number of GPUs for training (Detectron2 launch)")
    p.add_argument("--train-name", default="fwt_training", help="Detectron2 dataset name for train")
    p.add_argument("--val-name", default="twt_val", help="Detectron2 dataset name for val")
    p.add_argument("--num-classes", type=int, default=-1, help="If <0, infer from train json categories")
    return p.parse_args()


def main_worker(args):
    # 1) Register datasets
    register_datasets(args.train_name, args.val_name, args.train_json, args.train_img, args.val_json, args.val_img)

    # 2) Determine number of classes
    if args.num_classes > 0:
        num_classes = args.num_classes
    else:
        num_classes = infer_num_classes_from_json(args.train_json)

    # Show classes info in logs
    meta = MetadataCatalog.get(args.train_name)
    thing_classes = getattr(meta, "thing_classes", None)
    if thing_classes:
        print(f"[INFO] Detected classes ({len(thing_classes)}): {thing_classes}")
    else:
        print(f"[INFO] Using num_classes from JSON: {num_classes}")

    # 3) Build config
    cfg = setup_cfg(
        output_dir=args.output,
        base_config=args.config,
        train_name=args.train_name,
        val_name=args.val_name,
        num_classes=num_classes,
        ims_per_batch=args.ims_per_batch,
        base_lr=args.base_lr,
        max_iter=args.max_iter,
        warmup_iters=args.warmup_iters,
        num_workers=args.num_workers,
        image_size_train=args.image_size_train,
        image_size_test=args.image_size_test,
        batch_size_per_image=args.batch_size_per_image,
        num_queries=args.num_queries,
        seed=args.seed,
    )

    # 4) Log & train
    default_setup(cfg, args)  # sets up writers, logging, etc.
    print("[INFO] Training from scratch with random initialization (MODEL.WEIGHTS='').")
    print(f"[INFO] Output dir: {cfg.OUTPUT_DIR}")
    print(f"[INFO] Using {torch.cuda.device_count()} CUDA device(s).")

    trainer = Mask2FormerInstanceTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()


def main():
    args = parse_args()
    launch(
        main_worker,
        args.num_gpus,
        num_machines=1,
        machine_rank=0,
        dist_url="auto",
        args=(args,),
    )


if __name__ == "__main__":
    main()
