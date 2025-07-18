
# out_base가 정의되지 않아 NameError가 발생할 수 있으므로 상단에 추가
out_base = "/Users/iujeong/0.local/4.slice"

# ===============================
# ✅ 모델 추론하여 _pred.npy 생성
# ===============================
import sys
import os
import time
import csv
sys.path.append("/Users/iujeong/0.local/6.modelling")
from core.dataset import MeningiomaDataset, AugmentedMeningiomaDataset
from model.UNet_3 import UNet_3
from core.config import CFG
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np



model_path = "/Users/iujeong/0.local/6.modelling/result/pth/250712_2_epoch_036_dice_0.7823_best.pth"  # 수정 가능
model = UNet_3(in_channels=1, out_channels=1)  # 모델 구조 맞게 수정
model.load_state_dict(torch.load(model_path, map_location="cpu"))
model.eval()

print("\n🎯 모델 추론 시작 (_pred.npy 생성)")
for group in ["train", "val", "test"]:
    npy_dir = os.path.join(out_base, f"s_{group}/npy")
    pred_dir = os.path.join(out_base, f"s_{group}_pred")
    os.makedirs(pred_dir, exist_ok=True)
    for file in sorted(os.listdir(npy_dir)):
        if not file.endswith("_img.npy"):
            continue
        img_path = os.path.join(npy_dir, file)
        img = np.load(img_path)
        input_tensor = torch.tensor(img).unsqueeze(0).unsqueeze(0).float()
        with torch.no_grad():
            output = model(input_tensor)
            pred = torch.sigmoid(output).squeeze().numpy()
        save_path = os.path.join(pred_dir, file.replace("_img.npy", "_pred.npy"))
        np.save(save_path, pred)
        print(f"✅ Saved: {save_path}")