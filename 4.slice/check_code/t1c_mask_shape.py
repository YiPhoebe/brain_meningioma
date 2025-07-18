import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np

# 파일 경로 바꿔줘
img_path = "/Users/iujeong/0.local/3.normalize/n_train/nii/BraTS-MEN-RT-0112-1_norm.nii.gz"
gtv_path = "/Users/iujeong/0.local/2.resample/r_train/BraTS-MEN-RT-0112-1_t1c_gtv_mask.nii.gz"
bet_path = "/Users/iujeong/0.local/2.resample/r_train/BraTS-MEN-RT-0112-1_t1c_bet_mask.nii.gz"

# Load
img = nib.load(img_path)
gtv = nib.load(gtv_path)
bet = nib.load(bet_path)

img_data = img.get_fdata()
gtv_data = gtv.get_fdata()
bet_data = bet.get_fdata()

# ✅ 같은 방향으로 맞추기 (주의: shape도 바뀔 수 있음)
img_data = nib.as_closest_canonical(img).get_fdata()
gtv_data = nib.as_closest_canonical(gtv).get_fdata()
bet_data = nib.as_closest_canonical(bet).get_fdata()

# GTV 있는 중심 z-slice 찾기
z_idx = int(np.round(np.mean(np.argwhere(gtv_data > 0)[:, 2])))

plt.figure(figsize=(12, 4))

# Raw image
plt.subplot(1, 3, 1)
plt.imshow(img_data[:, :, z_idx], cmap="gray")
plt.title(f"Image Slice z={z_idx}")

# GTV overlaid
plt.subplot(1, 3, 2)
plt.imshow(img_data[:, :, z_idx], cmap="gray")
plt.imshow(gtv_data[:, :, z_idx], cmap="Reds", alpha=0.4)
plt.title("GTV Overlay")

# BET mask overlaid
plt.subplot(1, 3, 3)
plt.imshow(img_data[:, :, z_idx], cmap="gray")
plt.imshow(bet_data[:, :, z_idx], cmap="Blues", alpha=0.3)
plt.title("BET Overlay")

plt.tight_layout()
plt.show()