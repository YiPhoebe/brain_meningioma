import os
import sys
sys.path.append("/Users/iujeong/0.local/6.modelling")
from datetime import datetime
import torch
from model.UNet_2 import UNet_2  # 또는 nnUNet 등
from core.config import CFG
from core.dataset import MeningiomaDataset  # 네가 정의한 Dataset
from core.utils import dice_score
import matplotlib.pyplot as plt


# ✅ 설정
BATCH_SIZE = 8
MODEL_PATH = "/Users/iujeong/0.local/6.modelling/result/pth/epoch_026_dice_0.7808_best.pth"
DEVICE = "mps" if torch.cuda.is_available() else "cpu"

now = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_DIR = "/Users/iujeong/0.local/6.modelling/result/logs"
os.makedirs(LOG_DIR, exist_ok=True)
LOG_PATH = os.path.join(LOG_DIR, f"test_log_{now}.txt")
CSV_DIR = "/Users/iujeong/0.local/6.modelling/result/csv"
os.makedirs(CSV_DIR, exist_ok=True)
CSV_PATH = os.path.join(CSV_DIR, f"test_log_{now}.csv")
thresholds = [0.3, 0.5, 0.7]

log_file = open(LOG_PATH, "w")
log_file.write(f"Model: UNet(in_channels=1, out_channels=1)\n")
log_file.write(f"Batch size: {BATCH_SIZE}\n")
log_file.write(f"Thresholds: {thresholds}\n")
log_file.write(f"Model path: {MODEL_PATH}\n")
log_file.write(f"Test data dir: {CFG.test_dir}\n")
log_file.write(f"Device: {DEVICE}\n\n")

#
# ✅ 데이터셋 로딩
test_dataset = MeningiomaDataset(CFG.test_dir, mode="test")

# --- PAIR CHECK 코드 추가 ---
import numpy as np
for i in range(3):
    print("[PAIR CHECK]")
    print("Image path:", test_dataset.image_paths[i])
    print("Mask  path:", test_dataset.mask_paths[i])

    m = np.load(test_dataset.mask_paths[i])
    print(f"[{i}] Mask unique values: {np.unique(m)}, Sum: {m.sum()}")
# --- END PAIR CHECK ---

from torch.utils.data import DataLoader
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ✅ 모델 불러오기
model = UNet_2(in_channels=1, out_channels=1)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model = model.to(DEVICE)
model.eval()

# ✅ 테스트 루프
dice_dict = {t: 0.0 for t in thresholds}
valid_slice_counts = {t: 0 for t in thresholds}
volume_stats = []  # 부피 비교용

with torch.no_grad():
    for images, masks in test_loader:
        images, masks = images.to(DEVICE), masks.to(DEVICE)
        outputs = model(images)
        # 🔥 Resize outputs to match masks if needed
        if outputs.shape[2:] != masks.shape[2:]:
            outputs = torch.nn.functional.interpolate(outputs, size=masks.shape[2:], mode="bilinear", align_corners=False)
        probs = torch.sigmoid(outputs)
        print(f"probs min/max: {probs.min().item():.4f} / {probs.max().item():.4f}")

        print("image shape:", images.shape)
        print("mask shape:", masks.shape)
        print("output shape:", outputs.shape)

        for t in thresholds:
            preds = probs > t
            print("preds shape:", preds.shape)

            if preds.shape != masks.shape:
                raise ValueError(f"[ERROR] Shape mismatch! preds: {preds.shape}, masks: {masks.shape}")

            # Optional: skip slices with empty GT masks
            if masks.sum() == 0:
                continue

            # 2D 부피 기록
            gt_volume = masks.sum().item()
            pred_volume = preds.sum().item()
            volume_stats.append((gt_volume, pred_volume))

            valid_slice_counts[t] += 1
            dice_dict[t] += dice_score(preds, masks).item()

log_msg = "[Test Dice Score]\n"
with open(CSV_PATH, "w") as f_csv:
    f_csv.write("threshold,dice_score\n")
    for t in thresholds:
        if valid_slice_counts[t] > 0:
            avg_dice = dice_dict[t] / valid_slice_counts[t]
        else:
            avg_dice = 0.0
        f_csv.write(f"{t:.1f},{avg_dice:.4f}\n")
        log_msg += f"  threshold {t:.1f}: Dice = {avg_dice:.4f}\n"

print(log_msg)
log_file.write(log_msg + "\n")
log_file.close()


for i in range(images.shape[0]):
    img = images[i, 0].cpu().numpy()
    msk = masks[i, 0].cpu().numpy()
    prd = (probs[i, 0] > 0.5).cpu().numpy()

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(img, cmap='gray')
    plt.title('Image')
    plt.subplot(1, 3, 2)
    plt.imshow(msk, cmap='gray')
    plt.title('GT')
    plt.subplot(1, 3, 3)
    plt.imshow(msk, cmap='Blues', alpha=0.4)  # GT
    plt.imshow(prd, cmap='Reds', alpha=0.4)   # Prediction
    plt.title('Prediction Overlay')

    save_dir = "/Users/iujeong/0.local/6.modelling/visualize"
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f"test_{i}.png"))
    plt.close()

# ✅ Threshold별 Dice 그래프 저장
import pandas as pd

df = pd.read_csv(CSV_PATH)

plt.figure(figsize=(8, 5))
plt.plot(df['threshold'], df['dice_score'], marker='o')
plt.xlabel("Threshold")
plt.ylabel("Dice Score")
plt.title("Dice Score by Threshold")
plt.grid(True)
plt.xticks(np.arange(0.1, 1.0, 0.1))
plt.ylim(0, 1)

graph_path = os.path.join(LOG_DIR, f"dice_curve_{now}.png")
plt.savefig(graph_path)
plt.close()
print(f"[Saved] Dice curve → {graph_path}")

# ✅ 종양 부피 일치도 계산
vol_df = pd.DataFrame(volume_stats, columns=["gt_volume", "pred_volume"])
vol_df["volume_error"] = (vol_df["pred_volume"] - vol_df["gt_volume"]) / vol_df["gt_volume"]
vol_df["volume_error_abs"] = abs(vol_df["volume_error"])

vol_csv_path = os.path.join(CSV_DIR, f"volume_stats_{now}.csv")
vol_df.to_csv(vol_csv_path, index=False)
print(f"[Saved] Volume stats → {vol_csv_path}")

# 볼륨 오차 시각화
plt.figure(figsize=(7, 5))
plt.hist(vol_df["volume_error"], bins=20, color="orange", edgecolor="black")
plt.axvline(0, color="black", linestyle="--")
plt.xlabel("(Pred - GT) / GT")
plt.title("Volume Error Distribution")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(LOG_DIR, f"volume_error_{now}.png"))
plt.close()

# ✅ 위치별 성능 정리용 csv 저장 (예시 placeholder)
# 이 부분은 실제 위치 정보와 매핑되는 구조로 수정 필요
# 예시 형식: BraTS-MEN-RT-0011-1_slice_067_img.npy → frontal, temporal 등

# 가짜 예시: 임시 위치 매핑 (파일명 일부 기준으로 랜덤 배정)
def get_location_from_filename(fname):
    if "0011" in fname:
        return "frontal"
    elif "0022" in fname:
        return "temporal"
    elif "0033" in fname:
        return "occipital"
    else:
        return "unknown"

# volume_stats와 test_dataset.files는 같은 순서라고 가정
location_data = []
for (gt, pred), f in zip(volume_stats, test_dataset.files):
    loc = get_location_from_filename(f)
    dice = 1 - abs(pred - gt) / (gt + pred + 1e-8)  # 유사 Dice 계산
    location_data.append((f, loc, gt, pred, dice))

loc_df = pd.DataFrame(location_data, columns=["filename", "location", "gt_volume", "pred_volume", "dice_like"])
loc_csv_path = os.path.join(CSV_DIR, f"location_stats_{now}.csv")
loc_df.to_csv(loc_csv_path, index=False)
print(f"[Saved] Location-based stats → {loc_csv_path}")