from pathlib import Path
import nibabel as nib
import pandas as pd
import numpy as np
import os
from glob import glob
from tqdm import tqdm
import matplotlib.pyplot as plt

# 세 경로 지정
base_paths = {
    "train": Path("/Users/iujeong/0.local/2.resample/r_train"),
    "val": Path("/Users/iujeong/0.local/2.resample/r_val"),
    "test": Path("/Users/iujeong/0.local/2.resample/r_test"),
}

shape_records = []

for split_name, folder in base_paths.items():
    for t1c_file in folder.glob("*_t1c_bet.nii.gz"):
        case_id = t1c_file.name.replace("_t1c_bet.nii.gz", "")
        bet_path = folder / f"{case_id}_t1c_bet_mask.nii.gz"
        gtv_path = folder / f"{case_id}_t1c_gtv_mask.nii.gz"

        try:
            t1c_shape = nib.load(t1c_file).shape
            bet_shape = nib.load(bet_path).shape if bet_path.exists() else None
            gtv_shape = nib.load(gtv_path).shape if gtv_path.exists() else None

            gtv_mask = nib.load(gtv_path).get_fdata() if gtv_path.exists() else None
            bet_mask = nib.load(bet_path).get_fdata() if bet_path.exists() else None
            t1c_data = nib.load(t1c_file).get_fdata()

            gtv_voxels = int(np.count_nonzero(gtv_mask)) if gtv_mask is not None else None
            bet_voxels = int(np.count_nonzero(bet_mask)) if bet_mask is not None else None
            t1c_voxels = int(np.count_nonzero(t1c_data))

            shape_records.append({
                "split": split_name,
                "case_id": case_id,
                "t1c_shape": t1c_shape,
                "bet_shape": bet_shape,
                "gtv_shape": gtv_shape,
                "gtv_voxels": gtv_voxels,
                "bet_voxels": bet_voxels,
                "t1c_voxels": t1c_voxels,
            })
        except Exception as e:
            shape_records.append({
                "split": split_name,
                "case_id": case_id,
                "error": str(e),
                "t1c_shape": None,
                "bet_shape": None,
                "gtv_shape": None,
                "gtv_voxels": None,
                "bet_voxels": None,
                "t1c_voxels": None,
            })

df = pd.DataFrame(shape_records)
df.to_csv("/Users/iujeong/0.local/2.resample/volume_shapes.csv", index=False)
print(df)

# 예시 crop_and_pad 함수 (실제 함수는 별도 구현 필요)
def crop_and_pad(img, x_min, x_max, y_min, y_max):
    # Crop
    cropped = img[y_min:y_max, x_min:x_max]
    # Pad if needed (예시, 실제 pad 방식에 따라 수정)
    return cropped

# bbox_df는 외부에서 불러온다고 가정
# bbox_df = pd.read_csv('bbox.csv')

# Slices 저장 함수
def save_slices(input_dir, output_dir, phase, bbox_df):
    nii_files = sorted(glob(os.path.join(input_dir, "*_norm.nii.gz")))
    os.makedirs(output_dir, exist_ok=True)
    npy_dir = os.path.join(output_dir, "npy")
    png_dir = os.path.join(output_dir, "png")
    os.makedirs(npy_dir, exist_ok=True)
    os.makedirs(png_dir, exist_ok=True)

    for norm_path in tqdm(nii_files):
        pid = os.path.basename(norm_path).replace("_norm.nii.gz", "")
        gtv_path = os.path.join(input_dir, f"{pid}_gtv_mask.nii.gz")
        bbox = bbox_df[(bbox_df["patient_id"] == pid) & (bbox_df["phase"] == phase)]
        if bbox.empty or not os.path.exists(gtv_path):
            continue

        x_min, x_max = int(bbox["x_min"]), int(bbox["x_max"])
        y_min, y_max = int(bbox["y_min"]), int(bbox["y_max"])

        image = nib.load(norm_path).get_fdata()
        mask = nib.load(gtv_path).get_fdata()

        for i in range(image.shape[2]):
            img_slice = image[:, :, i]
            mask_slice = mask[:, :, i]

            cropped_img = crop_and_pad(img_slice, x_min, x_max, y_min, y_max)
            cropped_mask = crop_and_pad(mask_slice, x_min, x_max, y_min, y_max)

            np.save(os.path.join(npy_dir, f"{pid}_{i:03d}_img.npy"), cropped_img)
            np.save(os.path.join(npy_dir, f"{pid}_{i:03d}_mask.npy"), cropped_mask)

            plt.imsave(os.path.join(png_dir, f"{pid}_{i:03d}.png"), cropped_img, cmap='gray')