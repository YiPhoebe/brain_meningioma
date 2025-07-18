import sys
import os
import time
import csv
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

# Note: Data augmentations are assumed to be handled within the dataset class.
# DiceLoss implementation
# TverskyLoss implementation

from core.dataset import MeningiomaDataset, AugmentedMeningiomaDataset
from model.AttentionUNet import AttentionUNet
from core.config import CFG

if CFG.model_name == "nnUNet":
    print(f"[INFO] Using model: {CFG.model_name}")
    from model.nnUNet import nnUNet
    model = nnUNet(in_channels=1, out_channels=1).to(CFG.device)
    print(CFG.device)

elif CFG.model_name == "AttentionUNet":
    model = AttentionUNet(img_ch=1, output_ch=1).to(CFG.device)

elif CFG.model_name == "UNet_3":
    from model.UNet_3 import UNet_3
    model = UNet_3(in_channels=1, out_channels=1).to(CFG.device)
    print(f"[INFO] Using model: {CFG.model_name}")

else:
    raise ValueError(f"모델 이름 없음: {CFG.model_name}")


# 하이퍼파라미터 설정 (config.py)
# === Dataset & DataLoader ===
if CFG.use_aug:
    train_dataset = AugmentedMeningiomaDataset(CFG.train_dir)
else:
    train_dataset = MeningiomaDataset(CFG.train_dir)
print(f"[INFO] Using Dataset: {train_dataset.__class__.__name__}")
val_dataset = MeningiomaDataset(CFG.val_dir)

# === Debug: Print mask sum stats ===
print("[DEBUG] Train mask sum stats:")
for i in range(min(5, len(train_dataset))):
    _, m = train_dataset[i]
    print(f"[Train {i}] mask sum: {m.sum().item():.2f}, unique: {m.unique()}")

print("[DEBUG] Val mask sum stats:")
for i in range(min(5, len(val_dataset))):
    _, m = val_dataset[i]
    print(f"[Val {i}] mask sum: {m.sum().item():.2f}, unique: {m.unique()}")

train_loader = DataLoader(train_dataset, batch_size=CFG.batch_size,
                          shuffle=True, num_workers=4)

val_loader = DataLoader(val_dataset, batch_size=CFG.batch_size,
                        shuffle=False, num_workers=4)


# === Loss Function ===
# DiceLoss implementation
class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        intersection = (pred_flat * target_flat).sum()
        dice_loss = 1 - (2. * intersection + self.smooth) / (pred_flat.sum() + target_flat.sum() + self.smooth)
        return dice_loss
    
# TverskyLoss implementation
class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.3, beta=0.7, smooth=1e-5):
        super(TverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        TP = (pred_flat * target_flat).sum()
        FP = ((1 - target_flat) * pred_flat).sum()
        FN = (target_flat * (1 - pred_flat)).sum()
        tversky = (TP + self.smooth) / (TP + self.alpha * FP + self.beta * FN + self.smooth)
        return 1 - tversky

criterion = TverskyLoss(alpha=0.3, beta=0.7)


# === Optimizer ===
if CFG.optimizer == "Adam":
    optimizer = torch.optim.Adam(model.parameters(), lr=CFG.lr)
elif CFG.optimizer == "SGD":
    optimizer = torch.optim.SGD(model.parameters(), lr=CFG.lr, momentum=0.9)
else:
    raise NotImplementedError(f"Unknown optimizer: {CFG.optimizer}")


# === Scheduler ===
if CFG.scheduler == "StepLR":
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
elif CFG.scheduler == "None":
    scheduler = None
else:
    raise NotImplementedError(f"Unknown scheduler: {CFG.scheduler}")


# === Training Loop ===
def train():
    best_metric = -1.0

    def init_weights(m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            nn.init.xavier_uniform_(m.weight)
        elif isinstance(m, nn.BatchNorm2d):
            m.eval()  # freeze BN layers
    # model.apply(init_weights)

    print(f"[INFO] Training on device: {CFG.device}")

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_txt_path = os.path.join("/Users/iujeong/0.local/6.modelling/result/logs", f"train_log_{timestamp}.txt")
    log_csv_path = os.path.join("/Users/iujeong/0.local/6.modelling/result/csv", f"train_log_{timestamp}.csv")
    if not os.path.exists(log_csv_path):
        with open(log_csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "train_loss", "val_loss", "dice", "thresh_0.3", "thresh_0.5", "thresh_0.7"])

    with open(log_txt_path, "w") as log_file:
        log_file.write(f"Model: {CFG.model_name}\n")
        log_file.write(f"Loss: {CFG.loss_name}, Optimizer: {CFG.optimizer}, Scheduler: {CFG.scheduler}\n")
        log_file.write(f"Batch size: {CFG.batch_size}, LR: {CFG.lr}\n")
        log_file.write(f"Thresholds: [0.3, 0.5, 0.7]\n")
        log_file.write(f"Epochs: {CFG.num_epochs}\n\n")

    for epoch in range(CFG.num_epochs):
        model.train()
        train_loss = 0.0

        epoch_start = time.time()

        expected_size = (CFG.img_height, CFG.img_width)

        for images, masks in tqdm(train_loader, desc=f"[Epoch {epoch+1}] Training", leave=False):
            images, masks = images.to(CFG.device), masks.to(CFG.device)

            outputs = model(images)
            if outputs.shape[-2:] != expected_size:
                outputs = F.interpolate(outputs, size=expected_size, mode="bilinear", align_corners=False)
            if masks.shape[-2:] != expected_size:
                masks = F.interpolate(masks, size=expected_size, mode="nearest")
            loss = criterion(outputs, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        # === Validation ===
        model.eval()
        val_loss = 0.0
        sum_dice_03, sum_dice_05, sum_dice_07 = 0.0, 0.0, 0.0
        num_batches = 0

        with torch.no_grad():
            for images, masks in tqdm(val_loader, desc=f"[Epoch {epoch+1}] Validation", leave=False):
                images, masks = images.to(CFG.device), masks.to(CFG.device)

                outputs = model(images)
                if outputs.shape[-2:] != expected_size:
                    outputs = F.interpolate(outputs, size=expected_size, mode="bilinear", align_corners=False)
                if masks.shape[-2:] != expected_size:
                    masks = F.interpolate(masks, size=expected_size, mode="nearest")
                loss = criterion(outputs, masks)
                val_loss += loss.item()

                preds_03 = torch.sigmoid(outputs) > 0.3
                preds_05 = torch.sigmoid(outputs) > 0.5
                preds_07 = torch.sigmoid(outputs) > 0.7
                intersection_03 = (preds_03 * masks).sum()
                intersection_05 = (preds_05 * masks).sum()
                intersection_07 = (preds_07 * masks).sum()
                dice_0_3 = (2. * intersection_03 + 1e-5) / (preds_03.sum() + masks.sum() + 1e-5)
                dice_0_5 = (2. * intersection_05 + 1e-5) / (preds_05.sum() + masks.sum() + 1e-5)
                dice_0_7 = (2. * intersection_07 + 1e-5) / (preds_07.sum() + masks.sum() + 1e-5)

                sum_dice_03 += dice_0_3.item()
                sum_dice_05 += dice_0_5.item()
                sum_dice_07 += dice_0_7.item()
                num_batches += 1

        avg_val_loss = val_loss / len(val_loader)
        avg_dice_03 = sum_dice_03 / num_batches
        avg_dice_05 = sum_dice_05 / num_batches
        avg_dice_07 = sum_dice_07 / num_batches
        avg_val_dice = avg_dice_05

        print(f"[Epoch {epoch+1}] Threshold 0.3 Dice: {avg_dice_03:.4f}")
        print(f"[Epoch {epoch+1}] Threshold 0.5 Dice: {avg_dice_05:.4f}")
        print(f"[Epoch {epoch+1}] Threshold 0.7 Dice: {avg_dice_07:.4f}")

        # Write logs to CSV
        with open(log_csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch + 1,
                round(avg_train_loss, 4),
                round(avg_val_loss, 4),
                round(avg_val_dice, 4),
                round(avg_dice_03, 4),
                round(avg_dice_05, 4),
                round(avg_dice_07, 4),
            ])

        print(f"[Epoch {epoch+1}] Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Dice: {avg_val_dice:.4f}")

        with open(log_txt_path, "a") as log_file:
            log_file.write(f"[Epoch {epoch+1}] Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | ")
            log_file.write(f"Dice 0.3: {avg_dice_03:.4f}, Dice 0.5: {avg_dice_05:.4f}, Dice 0.7: {avg_dice_07:.4f}\n")


        # === Save Best Model ===
        if CFG.save_best_metric == "dice" and avg_val_dice > best_metric:
            best_metric = avg_val_dice
            save_name = f"250712_2_epoch_{epoch+1:03d}_dice_{avg_val_dice:.4f}_best.pth"
            torch.save(model.state_dict(), os.path.join(CFG.save_pth_dir, save_name))
            print(f"[INFO] Saved: {save_name}")

        elif CFG.save_best_metric == "val_loss" and avg_val_loss < best_metric:
            best_metric = avg_val_loss
            save_name = f"250712_2_epoch_{epoch+1:03d}_dice_{avg_val_dice:.4f}_best.pth"
            torch.save(model.state_dict(), os.path.join(CFG.save_pth_dir, save_name))
            print(f"[INFO] Saved: {save_name}")

        if scheduler is not None:
            scheduler.step()

        epoch_end = time.time()
        print(f"[Epoch {epoch+1}] Time taken: {(epoch_end - epoch_start):.2f} sec")


# === Main block ===
if __name__ == "__main__":
    train()
