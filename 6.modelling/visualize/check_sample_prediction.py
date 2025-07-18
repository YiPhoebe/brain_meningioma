import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))  # 상위 폴더 경로 추가
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F

from core.dataset import MeningiomaDataset
from model.nnUNet import nnUNet
from model.AttentionUNet import AttentionUNet
from core.config import CFG

# 데이터셋 로드
val_dataset = MeningiomaDataset(CFG.val_dir)

# 모델 정의 및 로드
if CFG.model_name == "nnUNet":
    model = nnUNet(in_channels=1, out_channels=1).to(CFG.device)
elif CFG.model_name == "AttentionUNet":
    model = AttentionUNet(img_ch=1, output_ch=1).to(CFG.device)

model.load_state_dict(torch.load(os.path.join(CFG.save_pth_dir, "best_model.pth"), map_location=CFG.device))
model.eval()

# 인덱스 0, 1번 샘플 확인
for idx in [0, 1]:
    image, mask = val_dataset[idx]
    image_tensor = image.unsqueeze(0).to(CFG.device)

    with torch.no_grad():
        output = model(image_tensor)
        output = F.interpolate(output, size=mask.shape[-2:], mode="bilinear", align_corners=False)
        pred = torch.sigmoid(output).squeeze().cpu().numpy()

    # 시각화
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(image.squeeze().cpu().numpy(), cmap="gray")
    plt.title("Input")

    plt.subplot(1, 3, 2)
    plt.imshow(mask.squeeze().cpu().numpy(), cmap="gray")
    plt.title("GT Mask")

    plt.subplot(1, 3, 3)
    plt.imshow(image.squeeze().cpu().numpy(), cmap="gray")
    plt.imshow(pred, cmap="Reds")
    plt.colorbar()
    plt.title("Prediction (Raw)")

    # Visualize thresholded predictions at multiple thresholds
    plt.figure(figsize=(15, 4))
    thresholds = [0.1, 0.3, 0.5, 0.7]
    for i, th in enumerate(thresholds):
        plt.subplot(1, len(thresholds), i + 1)
        plt.imshow(image.squeeze().cpu().numpy(), cmap="gray")
        plt.imshow(pred > th, cmap="Reds", alpha=0.5)
        plt.title(f"Threshold > {th}")
        plt.axis("off")
    plt.suptitle(f"Thresholding for Sample {idx}")
    plt.tight_layout()
    plt.show()

    plt.suptitle(f"Sample {idx}")
    plt.axis("off")
    plt.tight_layout()
    plt.show()