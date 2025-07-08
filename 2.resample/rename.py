import os
from glob import glob

# 경로는 r_train, r_val, r_test 다 포함
base_dirs = [
    "/Users/iujeong/0.local/2.resample/r_train",
    "/Users/iujeong/0.local/2.resample/r_val",
    "/Users/iujeong/0.local/2.resample/r_test",
]

for base_dir in base_dirs:
    files = sorted(glob(os.path.join(base_dir, "*_t1c_t1c_gtv_mask.nii.gz")))
    for old_path in files:
        new_path = old_path.replace("_t1c_t1c_gtv_mask.nii.gz", "_t1c_gtv_mask.nii.gz")
        if not os.path.exists(new_path):
            os.rename(old_path, new_path)
            print(f"✅ Renamed:\n  {old_path}\n→ {new_path}")
        else:
            print(f"⚠️ Already exists: {new_path} — skipped")