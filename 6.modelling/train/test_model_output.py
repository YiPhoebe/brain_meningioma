# test_model_output.py

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import torch
import torch.nn as nn
from core.config import CFG
from core.dataset import AugmentedMeningiomaDataset
from model.UNet_3 import UNet_3

# 모델 로딩
model = UNet_3().to(CFG.device)
model.eval()

# 데이터셋 하나 로딩
dataset = AugmentedMeningiomaDataset(CFG.val_dir)
img, mask = dataset[0]
img = img.unsqueeze(0).to(CFG.device)  # [1, 1, H, W]

# 추론
with torch.no_grad():
    output = model(img)
    pred_prob = torch.sigmoid(output)
    print(f"Sigmoid Output max: {pred_prob.max().item():.4f}, min: {pred_prob.min().item():.4f}")