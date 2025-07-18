from pathlib import Path
import pandas as pd
import numpy as np

# 세 경로 지정
base_paths = {
    "train": Path("/Users/iujeong/0.local/4.slice/s_train"),
    "val": Path("/Users/iujeong/0.local/4.slice/s_val"),
    "test": Path("/Users/iujeong/0.local/4.slice/s_test"),
}

shape_records = []

for split_name, folder in base_paths.items():
    for img_file in (folder / "npy").glob("*_img.npy"):
        case_id = img_file.name.replace("_img.npy", "")
        mask_file = img_file.parent / f"{case_id}_mask.npy"

        try:
            img = np.load(img_file)
            mask = np.load(mask_file) if mask_file.exists() else None

            img_shape = img.shape
            mask_shape = mask.shape if mask is not None else None
            mask_voxels = int(np.count_nonzero(mask)) if mask is not None else None
            img_voxels = int(np.count_nonzero(img))

            shape_records.append({
                "split": split_name,
                "case_id": case_id,
                "img_shape": img_shape,
                "mask_shape": mask_shape,
                "mask_voxels": mask_voxels,
                "img_voxels": img_voxels,
            })
        except Exception as e:
            shape_records.append({
                "split": split_name,
                "case_id": case_id,
                "error": str(e),
                "img_shape": None,
                "mask_shape": None,
                "mask_voxels": None,
                "img_voxels": None,
            })

df = pd.DataFrame(shape_records)
df.to_csv("/Users/iujeong/0.local/4.slice/npy_shapes.csv", index=False)
print(df)