
from glob import glob
import os

for phase in ["r_test", "r_train", "r_val"]:
    path = f"/Users/iujeong/0.local/2.resample/{phase}"
    files = glob(os.path.join(path, "*_t1c_bet.nii.gz"))
    patient_ids = set(os.path.basename(f).replace("_t1c_bet.nii.gz", "") for f in files)
    print(f"{phase}: {len(patient_ids)}명")