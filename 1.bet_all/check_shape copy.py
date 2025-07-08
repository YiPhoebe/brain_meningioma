

import os
import nibabel as nib
import pandas as pd
from glob import glob

def get_shape(filepath):
    if os.path.exists(filepath):
        try:
            return nib.load(filepath).shape
        except Exception as e:
            return f"Error: {e}"
    else:
        return "Missing"

def check_shapes(base_dirs, output_csv):
    rows = []

    for base_dir in base_dirs:
        nii_files = sorted(glob(os.path.join(base_dir, "*_bet.nii.gz")))

        for bet_path in nii_files:
            pid = os.path.basename(bet_path).replace("_t1c_bet.nii.gz", "")
            bet_mask_path = os.path.join(base_dir, f"{pid}_t1c_bet_mask.nii.gz")
            gtv_mask_path = os.path.join(base_dir, f"{pid}_t1c_gtv_mask.nii.gz")

            row = {
                "patient_id": pid,
                "bet_shape": get_shape(bet_path),
                "bet_mask_shape": get_shape(bet_mask_path),
                "gtv_mask_shape": get_shape(gtv_mask_path),
            }
            rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)
    print("✅ Saved to bet_mask_gtv_shapes.csv")

if __name__ == "__main__":
    base_dirs = [
        "/Users/iujeong/0.local/bet_all_old/b_test",
        "/Users/iujeong/0.local/bet_all_old/b_train",
        "/Users/iujeong/0.local/bet_all_old/b_val"
    ]
    check_shapes(base_dirs, "/Users/iujeong/0.local/bet_all_old/bet_mask_gtv_shapes.csv")