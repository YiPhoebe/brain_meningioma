import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import os

# 파일 경로 설정

slice_idx = 60

original_img_path = "/Users/iujeong/0.local/1.bet_all/b_train/BraTS-MEN-RT-0060-1_t1c_bet.nii.gz"
normalized_img_path = "/Users/iujeong/0.local/4.slice/s_train/npy/BraTS-MEN-RT-0060-1_slice_077_img.npy"
save_path = "/Users/iujeong/0.local/visualize/slice/slicebefore_after_slice_077.png"

# 파일 불러오기
original_img = nib.load(original_img_path).get_fdata()
norm_img = np.load(normalized_img_path)

# shape 및 슬라이스 추출
original_slice = original_img[:, :, slice_idx]
norm_slice = norm_img  # npy는 2D 슬라이스 이미지임

# 시각화
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
# 좌우 반전 제거 (종양이 왼쪽에 있도록 유지)
# original_slice = np.fliplr(original_slice)
# norm_slice = np.fliplr(norm_slice)

axes[0].imshow(np.rot90(np.fliplr(original_slice)), cmap='gray', origin='lower')
axes[0].set_title('Original MRI Slice')
axes[0].axis('off')

axes[1].imshow(np.rot90(np.fliplr(norm_slice)), cmap='gray', origin='lower')
axes[1].set_title('Normalized Slice')
axes[1].axis('off')

plt.tight_layout()
plt.savefig(save_path, dpi=300)
plt.close()
print(f"Saved to {save_path}")

# npy 슬라이스가 원본 nii의 몇 번째 슬라이스인지 찾기 (shape 일치할 때만 비교)
for i in range(original_img.shape[2]):
    orig_slice = original_img[:, :, i]
    if orig_slice.shape == norm_slice.shape:
        if np.allclose(norm_slice, orig_slice):
            print(f"정규화된 npy 슬라이스는 원본의 {i}번째 슬라이스와 동일합니다.")
            break
else:
    print("일치하는 슬라이스를 찾을 수 없습니다.")