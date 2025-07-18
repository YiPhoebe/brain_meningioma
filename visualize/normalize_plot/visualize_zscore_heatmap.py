import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
import os

def visualize_zscore_heatmap(nifti_path, bet_mask_path, save_path=None, slice_idx=None):
    # Load normalized image and BET mask
    img = nib.load(nifti_path).get_fdata()
    mask = nib.load(bet_mask_path).get_fdata()

    # Apply brain mask
    brain = img * (mask > 0)

    # Pick a slice
    if slice_idx is None:
        for i in range(img.shape[2]):
            if np.sum(mask[:, :, i]) > 500:  # 최소 픽셀 수 조건 추가
                slice_idx = i
                break
        else:
            print("No valid brain region found in any slice.")
            return

    print(f"Image shape: {img.shape}")
    print(f"Mask shape: {mask.shape}")
    print(f"Requested slice index: {slice_idx}")
    print(f"Nonzero mask count at slice: {np.sum(mask[:, :, slice_idx])}")

    # Compute mean and std only on brain pixels
    masked = mask[:, :, slice_idx] > 0
    slice_img = img[:, :, slice_idx]
    brain_pixels = slice_img[masked]

    mean = brain_pixels.mean()
    std = brain_pixels.std()

    if std == 0:
        print("Standard deviation is zero. Cannot compute Z-score.")
        return

    # Z-score만 계산 (배경은 NaN으로)
    zscore_slice = np.full(slice_img.shape, np.nan)
    zscore_slice[masked] = (slice_img[masked] - mean) / std

    # Mask NaN for visualization
    masked_zscore = np.ma.masked_invalid(zscore_slice)

    # Plot
    plt.figure(figsize=(6, 6))
    plt.imshow(masked_zscore, cmap='coolwarm', vmin=-3, vmax=3)
    plt.colorbar(label='Z-score')
    plt.title('Z-score Heatmap (Slice {})'.format(slice_idx))
    plt.axis('off')

    # Save or show
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
    else:
        plt.show()

# Example usage
if __name__ == "__main__":
    nifti_file = "/Users/iujeong/0.local/3.normalize/n_train/BraTS-MEN-RT-0060-1_norm.nii.gz"
    bet_mask_file = "/Users/iujeong/0.local/3.normalize/n_train/BraTS-MEN-RT-0060-1_bet_mask.nii.gz"
    save_file = "/Users/iujeong/0.local/visualize/normalize_plot/zscore_heatmap.png"
    visualize_zscore_heatmap(nifti_file, bet_mask_file, save_file)