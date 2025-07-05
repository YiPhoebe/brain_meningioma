
from glob import glob
import os

for phase in ["s_test", "s_train", "s_val"]:
    path = f"/Users/iujeong/0.local/4.slice/{phase}/npy"
    files = glob(os.path.join(path, "*_img.npy"))
    patient_ids = set(os.path.basename(f).split("_slice_")[0] for f in files)
    print(f"{phase}: {len(patient_ids)}명")