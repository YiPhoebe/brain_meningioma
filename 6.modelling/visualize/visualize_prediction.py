# visualize_prediction.py
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import torch
import matplotlib.pyplot as plt
from core.config import CFG
from core.dataset import MeningiomaDataset
from model.UNet_3 import UNet_3  # ← 네 모델에 따라 바꿔

model = UNet_3().to(CFG.device)
model.load_state_dict(torch.load("/Users/iujeong/0.local/6.modelling/result/pth/epoch_001_dice_0.1061_best.pth", map_location=CFG.device))
model.eval()

dataset = MeningiomaDataset(CFG.val_dir)
img, mask = dataset[0]  # [1, H, W]
img_input = img.unsqueeze(0).to(CFG.device)  # [1, 1, H, W]

with torch.no_grad():
    output = model(img_input)
    pred = torch.sigmoid(output).squeeze().cpu().numpy()

image = img.squeeze().cpu().numpy()
mask = mask.squeeze().cpu().numpy()

plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.title("Input")
plt.imshow(image, cmap="gray")

plt.subplot(1, 3, 2)
plt.title("GT Mask")
plt.imshow(mask, cmap="gray")

plt.subplot(1, 3, 3)
plt.title("Predicted Prob")
plt.imshow(pred, cmap="hot")
plt.colorbar()

plt.tight_layout()
plt.show()