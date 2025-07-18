import os
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

# 경로 수정
bet_img_path = "/Users/iujeong/0.local/1.bet_all/b_train/BraTS-MEN-RT-0060-1_t1c_bet.nii.gz"
bet_mask_path = "/Users/iujeong/0.local/1.bet_all/b_train/BraTS-MEN-RT-0060-1_t1c_bet_mask.nii.gz"
output_path = "/Users/iujeong/0.local/visualize/mask_overlay/bet_vs_mask_overlay.png"

# 이미지 불러오기
img = nib.load(bet_img_path).get_fdata()
mask = nib.load(bet_mask_path).get_fdata()
img = (img - img.min()) / (img.max() - img.min())

# 마스크 있는 중간 슬라이스
nonzero_slices = [i for i in range(mask.shape[2]) if mask[:, :, i].sum() > 0]
slice_idx = nonzero_slices[len(nonzero_slices) // 2]

# 시각화
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

axes[0].imshow(img[:, :, slice_idx], cmap="gray")
axes[0].set_title("BET Image")
axes[0].axis("off")

axes[1].imshow(img[:, :, slice_idx], cmap="gray")
axes[1].imshow(np.ma.masked_where(mask[:, :, slice_idx] == 0, mask[:, :, slice_idx]), cmap="autumn", alpha=0.4)
axes[1].set_title("BET + Mask Overlay")
axes[1].axis("off")

plt.tight_layout()
plt.savefig(output_path, bbox_inches="tight")
plt.close()