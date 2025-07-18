import numpy as np

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

            img_stats = {
                "img_min": float(img.min()),
                "img_max": float(img.max()),
                "img_mean": float(img.mean()),
                "img_std": float(img.std()),
                "img_unique": int(len(np.unique(img)))
            }

            if mask is not None:
                mask_stats = {
                    "mask_min": float(mask.min()),
                    "mask_max": float(mask.max()),
                    "mask_mean": float(mask.mean()),
                    "mask_std": float(mask.std()),
                    "mask_unique": int(len(np.unique(mask)))
                }
            else:
                mask_stats = {
                    "mask_min": None,
                    "mask_max": None,
                    "mask_mean": None,
                    "mask_std": None,
                    "mask_unique": None
                }

            shape_records.append({
                "split": split_name,
                "case_id": case_id,
                "img_shape": img_shape,
                "mask_shape": mask_shape,
                "mask_voxels": mask_voxels,
                "img_voxels": img_voxels,
                **img_stats,
                **mask_stats
            })

        except Exception as e:
            print(f"Error processing {img_file}: {e}")

df = pd.DataFrame(shape_records)
df.to_csv("/Users/iujeong/0.local/4.slice/npy_value.csv", index=False)
print(df)
