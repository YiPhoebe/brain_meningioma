import os
import nibabel as nib
import csv
from glob import glob

# 경로 설정
BASE_DIRS = [
    "/Users/iujeong/0.local/2.resample/r_test",
    "/Users/iujeong/0.local/2.resample/r_train",
    "/Users/iujeong/0.local/2.resample/r_val"
]

# 저장 경로
CSV_PATH = "/Users/iujeong/0.local/8.result/csv/pixel_spacing_info.csv"

# 저장할 리스트
spacing_data = []

# 확장자 기준
TARGET_SUFFIXES = ['_t1c_bet.nii.gz', '_t1c_bet_mask.nii.gz', '_t1c_gtv_mask.nii.gz']

for base_dir in BASE_DIRS:
    nii_files = sorted(glob(os.path.join(base_dir, "*.nii.gz")))

    for fpath in nii_files:
        if any(fpath.endswith(suffix) for suffix in TARGET_SUFFIXES):
            try:
                img = nib.load(fpath)
                spacing = img.header.get_zooms()[:3]
                spacing_data.append({
                    "filename": os.path.basename(fpath),
                    "path": fpath,
                    "spacing_x": spacing[0],
                    "spacing_y": spacing[1],
                    "spacing_z": spacing[2]
                })
            except Exception as e:
                print(f"❌ 에러 발생: {fpath} - {e}")

# CSV 저장
with open(CSV_PATH, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["filename", "path", "spacing_x", "spacing_y", "spacing_z"])
    writer.writeheader()
    writer.writerows(spacing_data)

print(f"✅ 저장 완료: {CSV_PATH} ({len(spacing_data)}개 파일)")