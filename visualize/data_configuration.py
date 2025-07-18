


import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np

# 파일 경로 설정
raw_img_path = "/Users/iujeong/0.local/original_data/all_t1c/BraTS-MEN-RT-0060-1_t1c.nii.gz"
bet_mask_path = "/Users/iujeong/0.local/1.bet_all/b_train/BraTS-MEN-RT-0060-1_t1c_bet_mask.nii.gz"
gtv_mask_path = "/Users/iujeong/0.local/1.bet_all/b_train/BraTS-MEN-RT-0060-1_t1c_gtv_mask.nii.gz"

# 이미지 로드
raw_img = nib.load(raw_img_path).get_fdata()
bet_mask = nib.load(bet_mask_path).get_fdata()
gtv_mask = nib.load(gtv_mask_path).get_fdata()

# BET 마스크가 존재하는 슬라이스 인덱스 중간값 선택
mid_slice = int(np.mean(np.nonzero(np.any(bet_mask, axis=(0, 1)))))

# 같은 슬라이스 추출
raw_slice = raw_img[:, :, mid_slice]
bet_slice = bet_mask[:, :, mid_slice]
gtv_slice = gtv_mask[:, :, mid_slice]

# 시각화
fig, axes = plt.subplots(1, 3, figsize=(9, 4))
axes[0].imshow(raw_slice.T, cmap='gray', origin='lower')
axes[0].set_title("Raw T1c Image")
axes[1].imshow(bet_slice.T, cmap='gray', origin='lower')
axes[1].set_title("BET Mask")
axes[2].imshow(gtv_slice.T, cmap='gray', origin='lower')
axes[2].set_title("GTV Mask")

for ax in axes:
    ax.axis('off')

plt.tight_layout(pad=1)
plt.subplots_adjust(wspace=0.1)
plt.savefig("/Users/iujeong/0.local/visualize/t1c_bet_gtv_row.png", dpi=300)
plt.show()