


import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from core.dataset import MeningiomaDataset  # 유정이 전처리 기반




# DiceLoss definition
class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super().__init__()
        self.smooth = smooth

    def forward(self, preds, targets):
        preds = preds.view(preds.size(0), -1)
        targets = targets.view(targets.size(0), -1)
        intersection = (preds * targets).sum(dim=1)
        dice = (2. * intersection + self.smooth) / (preds.sum(dim=1) + targets.sum(dim=1) + self.smooth)
        return 1 - dice.mean()

# Dice coefficient metric
def dice_coefficient(preds, targets, smooth=1e-5):
    preds = (preds > 0.5).float()
    preds = preds.view(preds.size(0), -1)
    targets = targets.view(targets.size(0), -1)
    intersection = (preds * targets).sum(dim=1)
    dice = (2. * intersection + smooth) / (preds.sum(dim=1) + targets.sum(dim=1) + smooth)
    return dice.mean().item()


 # UNet model definition
class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()

        def conv_block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            )

        self.encoder1 = conv_block(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)

        self.encoder2 = conv_block(64, 128)
        self.pool2 = nn.MaxPool2d(2)

        self.bottleneck = conv_block(128, 256)

        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.decoder2 = conv_block(256, 128)

        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder1 = conv_block(128, 64)

        self.conv_last = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        e1 = self.encoder1(x)
        e2 = self.encoder2(self.pool1(e1))
        b = self.bottleneck(self.pool2(e2))
        d2 = self.decoder2(torch.cat([self.up2(b), e2], dim=1))
        d1 = self.decoder1(torch.cat([self.up1(d2), e1], dim=1))
        return torch.sigmoid(self.conv_last(d1))

class MeningiomaTrainer:
    def __init__(self, config):
        self.config = config
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        print(f"[DEVICE] Using device: {self.device}")

        # Dataset
        train_dataset = MeningiomaDataset(config.train_dir)
        val_dataset = MeningiomaDataset(config.val_dir)

        self.train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

        # Model
        self.model = UNet().to(self.device)

        # Loss & Optimizer
        self.criterion = DiceLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.lr)

        # Checkpoint dir
        os.makedirs(config.save_dir, exist_ok=True)

    def train(self):
        for epoch in range(1, self.config.num_epochs + 1):
            self.model.train()
            epoch_loss = 0.0

            for x, y in tqdm(self.train_loader, desc=f"[Train] Epoch {epoch}"):
                x, y = x.to(self.device), y.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(x)
                loss = self.criterion(output, y)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()

            val_dice = self.validate()
            print(f"Epoch {epoch}: Train Loss: {epoch_loss:.4f}, Val Dice: {val_dice:.4f}")
            self.save_model(epoch)

    def validate(self):
        self.model.eval()
        dice_scores = []

        with torch.no_grad():
            for x, y in self.val_loader:
                x, y = x.to(self.device), y.to(self.device)
                output = self.model(x)
                dice = dice_coefficient(output, y)
                dice_scores.append(dice)

        return sum(dice_scores) / len(dice_scores)

    def save_model(self, epoch):
        save_path = os.path.join(self.config.save_dir, f"model_epoch{epoch}.pth")
        torch.save(self.model.state_dict(), save_path)


if __name__ == "__main__":
    from core.config import CFG
    trainer = MeningiomaTrainer(CFG)
    trainer.train()