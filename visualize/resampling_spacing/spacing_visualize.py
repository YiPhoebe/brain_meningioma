

import os
import nibabel as nib
import matplotlib.pyplot as plt

# 파일 경로 설정
original_path = '/Users/iujeong/0.local/original_data/all_t1c/BraTS-MEN-RT-0060-1_t1c.nii.gz'
resampled_path = '/Users/iujeong/0.local/2.resample/r_train/BraTS-MEN-RT-0060-1_t1c_bet.nii.gz'
save_path = '/Users/iujeong/0.local/visualize/resampling_spacing/spacing_comparison.png'

# NIfTI 파일 로드
original_nii = nib.load(original_path)
resampled_nii = nib.load(resampled_path)

# spacing 정보 추출
original_spacing = original_nii.header.get_zooms()
resampled_spacing = resampled_nii.header.get_zooms()

# 시각화
fig, ax = plt.subplots(figsize=(6, 4))
bar_labels = ['X', 'Y', 'Z']
x = range(3)
width = 0.35

ax.bar([i - width / 2 for i in x], original_spacing, width, label='Original')
ax.bar([i + width / 2 for i in x], resampled_spacing, width, label='Resampled')

ax.set_ylabel('Voxel Spacing (mm)')
ax.set_title('Voxel Spacing Comparison (Original vs Resampled)')
ax.set_xticks(x)
ax.set_xticklabels(bar_labels)
ax.legend()

plt.tight_layout()
plt.savefig(save_path, dpi=300)
plt.close()