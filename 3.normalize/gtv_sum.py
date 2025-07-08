

import os
import nibabel as nib
import csv
from glob import glob

# 입력 디렉토리 목록
input_dirs = {
    "train": "/Users/iujeong/0.local/3.normalize/n_train",
    "val": "/Users/iujeong/0.local/3.normalize/n_val",
    "test": "/Users/iujeong/0.local/3.normalize/n_test"
}

# 출력 CSV 경로
csv_path = "/Users/iujeong/0.local/8.result/csv/gtv_sum.csv"

# 결과 저장 리스트
results = []

# 각 phase 별로 반복
for phase, dir_path in input_dirs.items():
    nii_files = sorted(glob(os.path.join(dir_path, "**/*gtv_mask.nii.gz"), recursive=True))
    for path in nii_files:
        try:
            mask = nib.load(path).get_fdata()
            total = mask.sum()
            pid = os.path.basename(path).split("_")[0]
            results.append({
                "patient_id": pid,
                "phase": phase,
                "gtv_sum": total,
                "path": path
            })
        except Exception as e:
            print(f"❌ 오류: {path} → {e}")

# CSV 저장
with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["patient_id", "phase", "gtv_sum", "path"])
    writer.writeheader()
    writer.writerows(results)

print(f"✅ 저장 완료: {csv_path} ({len(results)}개)")