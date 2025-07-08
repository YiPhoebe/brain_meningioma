import os
import glob

base_path = "/Users/iujeong/0.local/2.resample"

for split in ["r_train", "r_val", "r_test"]:
    folder = os.path.join(base_path, split)
    files = sorted(glob.glob(os.path.join(folder, "*.nii.gz")))  # 또는 *.npy

    patient_ids = sorted(set([os.path.basename(f).split("_")[0] for f in files]))

    print(f"📁 {split} - 총 {len(patient_ids)}명:")
    for pid in patient_ids:
        print(pid)
    print("-" * 40)

import csv

output_csv = os.path.join(base_path, "patient_ids_by_split.csv")

with open(output_csv, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["split", "patient_id"])
    for split in ["r_train", "r_val", "r_test"]:
        folder = os.path.join(base_path, split)
        files = sorted(glob.glob(os.path.join(folder, "*.nii.gz")))
        patient_ids = sorted(set([os.path.basename(f).split("_")[0] for f in files]))
        for pid in patient_ids:
            writer.writerow([split, pid])

print(f"✅ CSV 저장 완료: {output_csv}")