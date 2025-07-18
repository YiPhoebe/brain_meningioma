# check_prediction_confidence.py

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import torch
from core.config import CFG
from core.dataset import MeningiomaDataset
from model.UNet_3 import UNet_3  # ← 모델에 따라 경로 바꿔줘

# 모델 로딩
model = UNet_3().to(CFG.device)
model.load_state_dict(torch.load("/Users/iujeong/0.local/6.modelling/result/pth/epoch_001_dice_0.0003.pth", map_location=CFG.device))
model.eval()

# 데이터 하나 로딩
dataset = MeningiomaDataset(CFG.val_dir)
img, mask = dataset[0]
img = img.unsqueeze(0).to(CFG.device)

# 예측
with torch.no_grad():
    output = model(img)
    prob = torch.sigmoid(output)
    print(f"Sigmoid max: {prob.max().item():.4f}")
    print(f"Sigmoid mean: {prob.mean().item():.4f}")