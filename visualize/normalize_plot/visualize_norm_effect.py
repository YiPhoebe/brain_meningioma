import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
import os

# 파일 경로 설정
before_path = '/Users/iujeong/0.local/2.resample/r_train/BraTS-MEN-RT-0060-1_t1c_bet.nii.gz'
after_path = '/Users/iujeong/0.local/3.normalize/n_train/BraTS-MEN-RT-0060-1_norm.nii.gz'
save_path = '/Users/iujeong/0.local/visualize/normalize_plot/comparison.png'

# 시각화할 슬라이스 index
slice_idx = 64  # 예시값, 원하는 위치로 바꿔

def load_nifti(path):
    return nib.load(path).get_fdata()

def plot_comparison(before_img, after_img, slice_idx, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    axes[0].imshow(before_img[:, :, slice_idx], cmap='gray')
    axes[0].set_title('Before Normalization')
    axes[0].axis('off')

    axes[1].imshow(after_img[:, :, slice_idx], cmap='gray')
    axes[1].set_title('After Normalization')
    axes[1].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

if __name__ == '__main__':
    before = load_nifti(before_path)
    after = load_nifti(after_path)
    plot_comparison(before, after, slice_idx, save_path)