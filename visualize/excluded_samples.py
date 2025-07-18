import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd

# padding 함수
def pad_to_square(img, target_size=None):
    h, w = img.shape
    if target_size is None:
        target_size = max(h, w)
    pad_h = (target_size - h) // 2
    pad_w = (target_size - w) // 2
    padded = np.pad(img, ((pad_h, target_size - h - pad_h),
                          (pad_w, target_size - w - pad_w)), mode='constant')
    return padded

# 입력 경로
bet_mask_path = "/Users/iujeong/0.local/1.bet_all/b_train/BraTS-MEN-RT-0005-1_t1c_bet_mask.nii.gz"
bet_image_path = "/Users/iujeong/0.local/1.bet_all/b_train/BraTS-MEN-RT-0005-1_t1c_bet.nii.gz"
gtv_mask_path = "/Users/iujeong/0.local/1.bet_all/b_train/BraTS-MEN-RT-0005-1_t1c_gtv_mask.nii.gz"

# 출력 경로
output_path = "/Users/iujeong/0.local/visualize/0005_overlay.png"

# 파일 로드
bet_img = nib.load(bet_image_path).get_fdata()
bet_mask = nib.load(bet_mask_path).get_fdata()
gtv_mask = nib.load(gtv_mask_path).get_fdata()


# BET 마스크로부터 직접 bounding box 계산
bet_mask_nonzero = np.argwhere(bet_mask > 0)
x_min, x_max = bet_mask_nonzero[:, 0].min(), bet_mask_nonzero[:, 0].max()
y_min, y_max = bet_mask_nonzero[:, 1].min(), bet_mask_nonzero[:, 1].max()

# GTV 마스크 중 BET 범위를 벗어나는 슬라이스 탐색
max_outside_distance = -1
target_slice = None

for z in range(gtv_mask.shape[2]):
    gtv = gtv_mask[:, :, z]
    if np.count_nonzero(gtv) == 0:
        continue
    gtv_coords = np.argwhere(gtv > 0)
    gx_min, gx_max = gtv_coords[:, 0].min(), gtv_coords[:, 0].max()
    gy_min, gy_max = gtv_coords[:, 1].min(), gtv_coords[:, 1].max()

    # 얼마나 BET 바깥에 떨어져 있는지 측정
    dx = max(x_min - gx_max, gx_min - x_max, 0)
    dy = max(y_min - gy_max, gy_min - y_max, 0)
    distance = dx + dy

    if distance > max_outside_distance:
        max_outside_distance = distance
        target_slice = z

# 3-Panel 시각화
# 시각화용 슬라이스 지정
z = target_slice if target_slice is not None else gtv_mask.shape[2] // 2
# GTV가 있는 axial 슬라이스에서 중심 위치 기준으로 sagittal/coronal 시각화
gtv_coords = np.argwhere(gtv_mask[:, :, z] > 0)
if len(gtv_coords) > 0:
    x = int(np.mean(gtv_coords[:, 0]))
    y = int(np.mean(gtv_coords[:, 1]))
else:
    x = gtv_mask.shape[0] // 2
    y = gtv_mask.shape[1] // 2

# 슬라이스 추출 (패딩 적용)
max_dim = max(bet_img.shape)
axial_bet = pad_to_square(bet_img[:, :, z], target_size=max_dim)
sagittal_bet = pad_to_square(bet_img[x, :, :], target_size=max_dim)
coronal_bet = pad_to_square(bet_img[:, y, :], target_size=max_dim)

axial_gtv = pad_to_square(gtv_mask[:, :, z], target_size=max_dim)
sagittal_gtv = pad_to_square(gtv_mask[x, :, :], target_size=max_dim)
coronal_gtv = pad_to_square(gtv_mask[:, y, :], target_size=max_dim)

# 시각화
fig, axes = plt.subplots(1, 3, figsize=(12, 5))

# Axial
axes[0].imshow(axial_bet.T, cmap='gray', origin='lower', vmin=0, vmax=1000)
axes[0].imshow(np.ma.masked_where(axial_gtv.T == 0, axial_gtv.T), cmap='autumn', alpha=0.5, origin='lower')
axes[0].set_title("Axial")
axes[0].axis('off')
axes[0].set_aspect('equal')

# Sagittal
axes[1].imshow(sagittal_bet.T, cmap='gray', origin='lower', vmin=0, vmax=1000)
axes[1].imshow(np.ma.masked_where(sagittal_gtv.T == 0, sagittal_gtv.T), cmap='autumn', alpha=0.5, origin='lower')
axes[1].set_title("Sagittal")
axes[1].axis('off')
axes[1].set_aspect('equal')

# Coronal
axes[2].imshow(coronal_bet.T, cmap='gray', origin='lower', vmin=0, vmax=1000)
axes[2].imshow(np.ma.masked_where(coronal_gtv.T == 0, coronal_gtv.T), cmap='autumn', alpha=0.5, origin='lower')
axes[2].set_title("Coronal")
axes[2].axis('off')
axes[2].set_aspect('equal')

plt.tight_layout()
plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.show()