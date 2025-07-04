import os
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from tqdm import tqdm
import nibabel as nib

def main():
    vis_dir = "/Users/iujeong/0.local/visualize"
    os.makedirs(vis_dir, exist_ok=True)

    input_dirs = [
        "/Users/iujeong/0.local/3.normalize/norm_test",
    ]
    for input_dir in input_dirs:
        img_paths = sorted(glob(os.path.join(input_dir, "*_norm.nii.gz")))
        for img_path in tqdm(img_paths, desc=f"Visualizing {os.path.basename(input_dir)}"):
            nii = nib.load(img_path)
            img = nii.get_fdata()

            print(f"👉 {os.path.basename(img_path)}: min={img.min():.3f}, max={img.max():.3f}, shape={img.shape}")

            base = os.path.basename(img_path).replace("_norm.nii.gz", "")
            for i in range(img.shape[2]):
                slice_img = img[:, :, i]
                plt.imsave(os.path.join(vis_dir, f"{base}_z{i:03}.png"), slice_img, cmap='gray', vmin=-3, vmax=3)

if __name__ == "__main__":
    main()