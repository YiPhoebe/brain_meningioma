import nibabel as nib
import numpy as np
import pandas as pd
from glob import glob
import os
from tqdm import tqdm


nii_files = sorted(glob("/Users/iujeong/0.local/3.normalize/n_*/*_norm.nii.gz"))

rows = []
for f in tqdm(nii_files):
    pid = os.path.basename(f).replace("_norm.nii.gz", "")
    phase = f.split("/")[-4].replace("n_", "")  # n_train → train 등

    img = nib.load(f).get_fdata()
    for z in range(img.shape[2]):
        slice_img = img[:, :, z]
        h, w = slice_img.shape
        rows.append({"patient_id": pid, "phase": phase, "z_index": z, "height": h, "width": w})

df = pd.DataFrame(rows)
df.to_csv("/Users/iujeong/0.local/8.result/csv/slice_shape_stats.csv", index=False)