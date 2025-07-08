import os
import nibabel as nib
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm

input_dirs = [
    "/Users/iujeong/0.local/2.resample/r_train",
    "/Users/iujeong/0.local/2.resample/r_val",
    "/Users/iujeong/0.local/2.resample/r_test",
]

results = []

for dir_path in input_dirs:
    nii_files = glob(os.path.join(dir_path, "*_mask.nii.gz"))
    for fpath in tqdm(nii_files, desc=f"Processing {os.path.basename(dir_path)}"):
        fname = os.path.basename(fpath)
        if "_gtv_mask" in fname:
            label = "GTV"
            image_fname = fname.replace("_gtv_mask.nii.gz", ".nii.gz")
        elif "_bet_mask" in fname:
            label = "BET"
            image_fname = fname.replace("_bet_mask.nii.gz", ".nii.gz")
        else:
            continue

        image_path = os.path.join(dir_path, image_fname)

        img = nib.load(fpath)
        data = img.get_fdata()
        voxel_count = int(np.sum(data > 0))

        orig_dir_path = dir_path.replace("2.resample", "1.bet_all").replace("r_", "b_")
        orig_path = os.path.join(orig_dir_path, fname)

        if os.path.exists(orig_path):
            orig_img = nib.load(orig_path)
            orig_data = orig_img.get_fdata()
            orig_voxel_count = int(np.sum(orig_data > 0))
        else:
            orig_voxel_count = -1  # marker for missing file

        removed_ratio = 1 - (voxel_count / orig_voxel_count) if orig_voxel_count > 0 else None

        results.append({
            "filename": fname,
            "image_file": image_fname,
            "type": label,
            "voxel_count": voxel_count,
            "original_voxel_count": orig_voxel_count,
            "removed_ratio": removed_ratio,
        })

df = pd.DataFrame(results)
save_path = "/Users/iujeong/0.local/2.resample/voxel_counts.csv"
df.to_csv(save_path, index=False)
print(f"✅ Voxel counts saved to {save_path}")

df["patient_id"] = df["filename"].str.extract(r'(BraTS-MEN-RT-\d{4}-\d)')
df_filtered = df.copy()

# Pivot so each patient has GTV/BET columns
df_grouped = df_filtered.pivot(index="patient_id", columns="type", values="removed_ratio").reset_index()
df_grouped = df_grouped.sort_values(by="patient_id").reset_index(drop=True)
df_grouped.to_csv("/Users/iujeong/0.local/2.resample/removed_ratio_over_40_grouped.csv", index=False)
print("⚠️ Saved patients with >40% voxel loss (BET or GTV).")