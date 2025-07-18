import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

# 경로 설정
before_path = "/Users/iujeong/0.local/2.resample/r_train/BraTS-MEN-RT-0060-1_t1c_bet.nii.gz"
after_path = "/Users/iujeong/0.local/3.normalize/n_train/BraTS-MEN-RT-0060-1_norm.nii.gz"
bet_mask_path = "/Users/iujeong/0.local/3.normalize/n_train/BraTS-MEN-RT-0060-1_bet_mask.nii.gz"
save_path = "/Users/iujeong/0.local/visualize/normalize_plot/intensity_distribution_comparison.png"

# 데이터 불러오기
before_img = nib.load(before_path).get_fdata()
after_img = nib.load(after_path).get_fdata()
mask = nib.load(bet_mask_path).get_fdata()

# 뇌 영역만 선택
brain_before = before_img[mask > 0]
brain_after = after_img[mask > 0]

# 히스토그램 시각화
plt.figure(figsize=(8, 5))
plt.hist(brain_before.flatten(), bins=100, alpha=0.5, color='skyblue', label='Before')
plt.hist(brain_after.flatten(), bins=100, alpha=0.8, color='sandybrown', label='After')
plt.yscale("log")
plt.axvline(x=0, color='r', linestyle='--', label='mean=0')
plt.axvline(x=1, color='g', linestyle='--')
plt.axvline(x=-1, color='g', linestyle='--')
plt.legend()
plt.title('Intensity Distribution (Before vs After Normalization)')
plt.xlabel('Intensity')
plt.ylabel('Voxel Count (log scale)')
plt.tight_layout()
plt.savefig(save_path, dpi=200)
plt.show()