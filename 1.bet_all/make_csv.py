

import os
import pandas as pd

input_dirs = {
    "test": "/Users/iujeong/0.local/1.bet_all/b_test",
    "train": "/Users/iujeong/0.local/1.bet_all/b_train",
    "val": "/Users/iujeong/0.local/1.bet_all/b_val",
}

output_dir = "/Users/iujeong/0.local/result/csv"
os.makedirs(output_dir, exist_ok=True)

def extract_patient_id(filename):
    return filename.split("_")[0]

for split, dir_path in input_dirs.items():
    files = [f for f in os.listdir(dir_path) if f.endswith(".nii.gz") and not f.startswith(".")]
    patient_ids = sorted(set(extract_patient_id(f) for f in files))
    df = pd.DataFrame(patient_ids, columns=["patient_id"])
    df.to_csv(os.path.join(output_dir, f"patients_{split}.csv"), index=False)
    print(f"Saved {split} patients to CSV: {len(patient_ids)} IDs")