import os
import numpy as np
from glob import glob
from tqdm import tqdm
import csv

csv_save_path = "/Users/iujeong/0.local/8.result/csv/npy_background_report.csv"
log_save_path = "/Users/iujeong/0.local/8.result/log/npy_background_check.log"

os.makedirs(os.path.dirname(csv_save_path), exist_ok=True)
os.makedirs(os.path.dirname(log_save_path), exist_ok=True)

report_rows = []

for root in [
    "/Users/iujeong/0.local/4.slice/s_train/npy",
    "/Users/iujeong/0.local/4.slice/s_val/npy",
    "/Users/iujeong/0.local/4.slice/s_test/npy"
]:
    print(f"\n🔍 Checking root: {root}")
    npy_paths = sorted(glob(os.path.join(root, "*.npy")))
    print(f"찾은 npy 개수: {len(npy_paths)}개")

    for path in tqdm(npy_paths):
        arr = np.load(path)

        nonzeros = arr[arr != 0]
        if nonzeros.size > 0:
            min_val = nonzeros.min()
        else:
            min_val = 0

        background_mask = (arr <= 0)
        background_values = arr[background_mask]

        print(f"\n📂 {os.path.basename(path)}")
        print(f"  0이 아닌 최소값: {min_val:.10f}")
        print(f"  배경값 최소: {background_values.min():.10f}")
        print(f"  배경값 최대: {background_values.max():.10f}")
        print(f"  배경 픽셀 수: {background_values.size}")

        report_rows.append({
            "filename": os.path.basename(path),
            "min_nonzero": f"{min_val:.10f}",
            "bg_min": f"{background_values.min():.10f}",
            "bg_max": f"{background_values.max():.10f}",
            "bg_pixel_count": background_values.size
        })

# CSV 저장
with open(csv_save_path, "w", newline="") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=["filename", "min_nonzero", "bg_min", "bg_max", "bg_pixel_count"])
    writer.writeheader()
    writer.writerows(report_rows)

# 로그 저장
with open(log_save_path, "w") as logfile:
    for row in report_rows:
        logfile.write(
            f"{row['filename']} | 0이 아닌 최소값: {row['min_nonzero']} | 배경 최소: {row['bg_min']} | 배경 최대: {row['bg_max']} | 배경 픽셀 수: {row['bg_pixel_count']}\n"
        )