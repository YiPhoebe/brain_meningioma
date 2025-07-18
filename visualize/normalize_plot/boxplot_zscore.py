import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

# 파일 경로 설정
img_path = "/Users/iujeong/0.local/3.normalize/n_train/BraTS-MEN-RT-0060-1_norm.nii.gz"
mask_path = "/Users/iujeong/0.local/3.normalize/n_train/BraTS-MEN-RT-0060-1_bet_mask.nii.gz"

# 이미지 & 마스크 불러오기
img_nii = nib.load(img_path)
mask_nii = nib.load(mask_path)

img = img_nii.get_fdata()
mask = mask_nii.get_fdata()

# 마스크 > 0인 뇌 영역 픽셀만 선택
brain_pixels = img[mask > 0]

# 박스플롯 그리기
plt.figure(figsize=(5.5, 5.5))
plt.boxplot(brain_pixels, vert=True, patch_artist=True,
            boxprops=dict(facecolor='skyblue', color='black'),
            medianprops=dict(color='red'),
            whiskerprops=dict(color='black'),
            capprops=dict(color='black'),
            flierprops=dict(markerfacecolor='orange', marker='o', markersize=5, linestyle='none'))

plt.title('Z-score Distribution (Brain Only)')
plt.ylabel('Z-score')
plt.xticks([1], ['Brain'])
plt.axhline(0, color='gray', linestyle='--', linewidth=1)

plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig("/Users/iujeong/0.local/visualize/normalize_plot/boxplot_zscore.png", dpi=150)
plt.show()