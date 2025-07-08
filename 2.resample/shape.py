from pathlib import Path
import nibabel as nib
import pandas as pd
import numpy as np

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

            gtv_voxels = int(np.count_nonzero(gtv_mask > 0.5)) if gtv_mask is not None else None
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