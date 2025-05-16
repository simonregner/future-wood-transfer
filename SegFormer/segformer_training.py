import os
from PIL import Image
from datasets import Dataset
from transformers import (
    SegformerForSemanticSegmentation,
    SegformerImageProcessor,
    TrainingArguments,
    Trainer
)
import torch

# --- CONFIGURATION ---
NUM_CLASSES = 8
image_dir = '/home/simon/Documents/Master-Thesis/data/yolo_training_data/train/images'
mask_dir = '/home/simon/Documents/Master-Thesis/data/yolo_training_data/train/masks'
output_dir = "./segformer_output"
os.makedirs(output_dir, exist_ok=True)


# --- LOAD DATASET ---
def load_dataset(image_dir, mask_dir):
    files = sorted(os.listdir(image_dir))
    images = []
    masks = []

    for file in files:
        if file.endswith(".jpg") or file.endswith(".png"):
            images.append(os.path.join(image_dir, file))
            masks.append(os.path.join(mask_dir, file.replace(".jpg", ".png")))

    return Dataset.from_dict({"image": images, "segmentation_mask": masks})


dataset = load_dataset(image_dir, mask_dir)

# --- PREPROCESSING ---
processor = SegformerImageProcessor(reduce_labels=False)


def preprocess(example):
    image = Image.open(example["image"]).convert("RGB")
    mask = Image.open(example["segmentation_mask"])
    inputs = processor(image, segmentation_maps=mask, return_tensors="pt")
    return {
        "pixel_values": inputs["pixel_values"].squeeze(),
        "labels": inputs["labels"].squeeze()
    }


dataset = dataset.map(preprocess)


split = dataset.train_test_split(test_size=0.2)  # 20% for validation
train_dataset = split["train"]
val_dataset = split["test"]

# --- INITIALIZE MODEL ---
model = SegformerForSemanticSegmentation.from_pretrained(
    "nvidia/segformer-b0-finetuned-ade-512-512",
    num_labels=NUM_CLASSES,
    ignore_mismatched_sizes=True
)

# --- TRAINING ---
args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=12,
    learning_rate=5e-5,
    num_train_epochs=10,
    save_strategy="epoch",
    logging_dir=f"{output_dir}/logs",
    logging_steps=10,
    report_to="none",  # avoids errors in non-W&B environments
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=processor
)

trainer.train()

# --- SAVE FINAL MODEL ---
model.save_pretrained(f"{output_dir}/final_model")
processor.save_pretrained(f"{output_dir}/final_model")