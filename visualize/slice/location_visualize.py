import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

def compute_iou(pred, gt):
    pred_bin = (pred > 0.5).astype(np.uint8)
    gt_bin = (gt > 0.5).astype(np.uint8)
    intersection = np.logical_and(pred_bin, gt_bin).sum()
    union = np.logical_or(pred_bin, gt_bin).sum()
    return intersection / union if union != 0 else 0.0

# 📌 불러오기
csv_path = "/Users/iujeong/0.local/4.slice/tumor_centers.csv"
csv = pd.read_csv(csv_path)

ious = []
for _, row in csv.iterrows():
    patient_id = row['PatientID']
    slice_idx = int(row['SliceIdx'])

    pred_path = f"/Users/iujeong/0.local/4.slice/s_train/npy/{patient_id}_slice_{slice_idx:03d}_pred.npy"
    gt_path = row['MaskPath']

    if os.path.exists(pred_path) and os.path.exists(gt_path):
        pred = np.load(pred_path)
        gt = np.load(gt_path)
        iou = compute_iou(pred, gt)
    else:
        iou = np.nan

    ious.append(iou)

csv['IoU'] = ious
csv.to_csv(csv_path, index=False)
print("Updated CSV with IoU values.")

# 예시: 특정 파일 하나만 처리해보자
patient_id = "BraTS-MEN-RT-0060-1"
slice_idx = 77
filename = f"/Users/iujeong/0.local/4.slice/s_train/npy/{patient_id}_slice_{slice_idx:03d}_img.npy"
img_path = filename

# NIfTI 이미지 로딩
volume = np.load(img_path)

# 해당 환자의 좌표들만 필터링
coords = csv[(csv['PatientID'] == patient_id) & (csv['SliceIdx'] == slice_idx)]

# 하나씩 시각화
for _, row in coords.iterrows():
    slice_img = volume
    x, y = float(row['X']), float(row['Y'])
    # y = slice_img.shape[0] - y

    plt.imshow(slice_img.T, cmap='gray', origin='lower')
    plt.scatter(y, x, c='red', s=40)  # 좌표 표시
    plt.title(f"{patient_id} - Slice {slice_idx}")
    plt.axis('off')
    save_path = f"/Users/iujeong/0.local/visualize/slice/{patient_id}_slice{slice_idx:03d}_coord.png"
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    print(f"Saved: {save_path}")
    plt.show()

import seaborn as sns

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# 1. 2D 히트맵 (axial slice 기준)
all_x = csv['X'].astype(float).values
all_y = csv['Y'].astype(float).values
hist = axes[0].hist2d(all_y, all_x, bins=50, cmap='hot')
axes[0].set_title("Tumor Center Heatmap (X vs Y)")
axes[0].set_xlabel("Y axis")
axes[0].set_ylabel("X axis")
cbar = plt.colorbar(hist[3], ax=axes[0])
cbar.set_label('Count')
print("📌 histogram plotted")

# 2. 부위별 막대그래프 (label 컬럼이 있을 경우)
if 'Region' in csv.columns:
    region_counts = csv['Region'].value_counts()
    sns.barplot(x=region_counts.values, y=region_counts.index, ax=axes[1])
    axes[1].set_title("Tumor Distribution by Region")
    axes[1].set_xlabel("Count")
    axes[1].set_ylabel("Region")
    axes[1].invert_yaxis()
else:
    axes[1].text(0.5, 0.5, 'No Region column found', ha='center', va='center')
    axes[1].set_axis_off()

# 3. X, Y, Z 좌표 분포 히스토그램
if 'Z' in csv.columns:
    axes[2].hist(csv[['X', 'Y', 'Z']].astype(float).values, bins=40, label=['X', 'Y', 'Z'])
    axes[2].legend()
    axes[2].set_title("X / Y / Z Coordinate Distribution")
    axes[2].set_xlabel("Value")
    axes[2].set_ylabel("Frequency")
else:
    axes[2].hist(csv[['X', 'Y']].astype(float).values, bins=40, label=['X', 'Y'], color=["#1f77b4", "#ff7f0e"])
    axes[2].legend()
    axes[2].set_title("X and Y Coordinate Histogram")
    axes[2].set_xlabel("Value")
    axes[2].set_ylabel("Frequency")

plt.tight_layout()
combo_path = "/Users/iujeong/0.local/visualize/slice/overview_subplot.png"
plt.savefig(combo_path, dpi=150)
print(f"Saved combined overview: {combo_path}")
plt.close()