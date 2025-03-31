import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.yolo_road_seg_model import YOLO12RoadSegModel

# -----------------------------
# Trainer
# -----------------------------
class RoadSegTrainer:
    def __init__(self, train_dataset, val_dataset=None, epochs=10, batch_size=2, lr=1e-4):
        self.model = None
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.epochs = epochs
        self.batch_size = batch_size
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False) if val_dataset else None
        self.lr = lr
        self.optimizer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.bce_loss = nn.BCEWithLogitsLoss()
        self.ce_loss = nn.CrossEntropyLoss()

    def get_model(self, cfg, num_classes=8, num_road_groups=10):
        self.model = YOLO12RoadSegModel(cfg=cfg, nc=num_classes, num_road_groups=num_road_groups)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.model.to(self.device)


    def train(self):
        for epoch in range(self.epochs):
            self.model.train()
            total_loss = 0
            for img, masks, class_ids, road_ids in tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.epochs} [Train]"):
                img = img.to(self.device)
                masks = masks.to(self.device)
                road_ids = road_ids.to(self.device)

                out = self.model(img)

                print(out)

                if isinstance(out, (list, tuple)) and len(out) == 2:
                    seg_out, group_out = out
                else:
                    raise ValueError(f"Expected 2 outputs from model, got {len(out)}")

                combined_mask = masks.sum(dim=1, keepdim=True).clamp(0, 1)
                road_id_target = road_ids[:, 0]
                road_group_logits = group_out.mean(dim=[2, 3])

                loss_seg = self.bce_loss(seg_out[0], combined_mask)
                loss_group = self.ce_loss(road_group_logits, road_id_target)
                loss = loss_seg + loss_group

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

            print(f"Epoch {epoch+1}: Train Loss = {total_loss / len(self.train_loader):.4f}")

            if self.val_loader:
                self.validate()

    def validate(self):
        self.model.eval()
        val_loss = 0
        with torch.no_grad():
            for img, masks, class_ids, road_ids in tqdm(self.val_loader, desc="Validation"):
                img = img.to(self.device)
                masks = masks.to(self.device)
                road_ids = road_ids.to(self.device)

                seg_out, group_out = self.model(img)

                combined_mask = masks.sum(dim=1, keepdim=True).clamp(0, 1)
                road_id_target = road_ids[:, 0]
                road_group_logits = group_out.mean(dim=[2, 3])

                loss_seg = self.bce_loss(seg_out[0], combined_mask)
                loss_group = self.ce_loss(road_group_logits, road_id_target)
                loss = loss_seg + loss_group

                val_loss += loss.item()

        print(f"Validation Loss = {val_loss / len(self.val_loader):.4f}")