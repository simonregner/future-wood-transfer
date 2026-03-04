from utils.yolo_seg_dataset import YoloTxtSegDataset
from utils.yolo_road_seg_model import YOLO12RoadSegModel
from utils.road_seg_trainer import RoadSegTrainer

# -----------------------------
# Run Training
# -----------------------------
if __name__ == "__main__":
    train_txt_dir = "/home/simon/Documents/Master-Thesis/data/yolo_training_data/train/labels"
    train_img_dir = "/home/simon/Documents/Master-Thesis/data/yolo_training_data/train/images"
    val_txt_dir = "/home/simon/Documents/Master-Thesis/data/yolo_training_data/val/labels"
    val_img_dir = "/home/simon/Documents/Master-Thesis/data/yolo_training_data/val/images"

    train_dataset = YoloTxtSegDataset(train_txt_dir, train_img_dir, image_size=(640, 640), include_classes=[7])
    val_dataset = YoloTxtSegDataset(val_txt_dir, val_img_dir, image_size=(640, 640), include_classes=[7])

    trainer = RoadSegTrainer(train_dataset, val_dataset=val_dataset, epochs=10, batch_size=2, lr=1e-4)
    trainer.get_model(cfg='yolo12-seg.yaml', num_classes=8, num_road_groups=10)
    trainer.train()