import os
import re
import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio
from glob import glob
from tqdm import tqdm
import nibabel as nib
from skimage.transform import resize

def main():
    vis_dir = "/Users/iujeong/0.local/5.npy_visualize"
    os.makedirs(vis_dir, exist_ok=True)

    input_dirs = [
        "/Users/iujeong/0.local/4.slice/s_train",
        "/Users/iujeong/0.local/4.slice/s_val",
        "/Users/iujeong/0.local/4.slice/s_test",
    ]

    output_dirs = {
        "s_train": "/Users/iujeong/0.local/5.npy_visualize/o_train",
        "s_val": "/Users/iujeong/0.local/5.npy_visualize/o_val",
        "s_test": "/Users/iujeong/0.local/5.npy_visualize/o_test",
    }

    for input_dir in input_dirs:
        img_paths = sorted(glob(os.path.join(input_dir, "*", "*_img.npy")))
        for img_path in tqdm(img_paths, desc=f"Visualizing {os.path.basename(input_dir)}"):
            img = np.load(img_path)

            base = os.path.basename(img_path).replace("_img.npy", "")
            match = re.match(r"(BraTS-MEN-RT-\d{4}-\d+)_z(\d{3})", base)
            if not match:
                print(f"[ERROR] Failed to parse patient_id and z_idx from base: {base}")
                continue
            patient_id = match.group(1)
            z_idx = int(match.group(2))
            print(f"[DEBUG] base={base}, patient_id={patient_id}, z_idx={z_idx}")

            bet_dir_map = {
                "s_train": "/Users/iujeong/0.local/2.resample/r_train",
                "s_val": "/Users/iujeong/0.local/2.resample/r_val",
                "s_test": "/Users/iujeong/0.local/2.resample/r_test",
            }
            subdir = os.path.basename(input_dir)
            bet_nii_path = os.path.join(bet_dir_map[subdir], f"{patient_id}_t1c_bet_mask.nii.gz")

            if os.path.exists(bet_nii_path):
                bet_nii = nib.load(bet_nii_path).get_fdata()
                bet_slice = bet_nii[:, :, z_idx]
                bet_slice_resized = resize(bet_slice, img.shape, order=0, preserve_range=True, anti_aliasing=False).astype(np.uint8)
                background_pixels = img[bet_slice_resized == 0]
                if np.any(background_pixels != 0):
                    with open("/Users/iujeong/0.local/5.npy_visualize/nonzero_background_log.csv", "a") as f:
                        f.write(f"{base},{np.count_nonzero(background_pixels)}\n")
                img[bet_slice_resized == 0] = 0.0
            else:
                print(f"[WARNING] BET NIfTI not found for {bet_nii_path}")

            base = os.path.basename(img_path).replace("_img.npy", "")
            print(f"📊 Full array for {base}:")
            print(img)

            print(f"👉 {os.path.basename(img_path)}: min={img.min():.3f}, max={img.max():.3f}, shape={img.shape}")

            subdir = os.path.basename(input_dir)
            save_root = output_dirs[subdir]
            os.makedirs(save_root, exist_ok=True)
            save_path = os.path.join(save_root, f"{base}.png")

            img_clipped = np.clip(img, -2, 2)
            img_rescaled = (img_clipped + 2) / 4  # scale to [0,1]
            img_uint8 = (img_rescaled * 255).astype(np.uint8)
            imageio.imwrite(save_path, img_uint8)

if __name__ == "__main__":
    main()