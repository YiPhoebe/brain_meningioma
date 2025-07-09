import os
import numpy as np
import nibabel as nib
from glob import glob

# 리샘플된 마스크 경로들
mask_dirs = [
    "/Users/iujeong/0.local/2.resample/r_test",
    "/Users/iujeong/0.local/2.resample/r_train",
    "/Users/iujeong/0.local/2.resample/r_val",
]

mask_paths = []
for d in mask_dirs:
    mask_paths.extend(sorted(glob(os.path.join(d, "*_mask.nii.gz"))))

log_path = "/Users/iujeong/0.local/2.resample/mask_value_log.csv"

with open(log_path, "w") as f:
    f.write("filename,unique_values\n")
    for path in mask_paths:
        nii = nib.load(path)
        data = nii.get_fdata()
        unique = np.unique(data)
        f.write(f"{os.path.basename(path)},{'|'.join(map(str, unique))}\n")