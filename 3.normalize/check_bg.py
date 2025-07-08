import os
import nibabel as nib
import numpy as np
from glob import glob
from tqdm import tqdm

# 검사할 루트 디렉토리들
root_dirs = [
    "/Users/iujeong/0.local/3.normalize/n_train",
    "/Users/iujeong/0.local/3.normalize/n_val",
    "/Users/iujeong/0.local/3.normalize/n_test",
]

# 배경이 0이 아닌 샘플들을 저장할 리스트
nonzero_background_samples = []

for root in root_dirs:
    print(f"🔍 Checking root: {root}", flush=True)
    # 여기 glob 수정!!
    norm_files = sorted(glob(os.path.join(root, "**", "*_norm.nii.gz"), recursive=True))
    print(f"{root} 안에서 찾은 파일 개수: {len(norm_files)}개", flush=True)
    for norm_path in tqdm(norm_files):
        base = os.path.basename(norm_path).replace("_norm.nii.gz", "")
        bet_mask_path = norm_path.replace("_norm.nii.gz", "_bet_mask.nii.gz")
        
        # 파일 존재 여부 체크
        if not os.path.exists(bet_mask_path):
            print(f"❌ BET mask not found: {bet_mask_path}", flush=True)
            continue

        # 파일 로딩
        norm_img = nib.load(norm_path).get_fdata()
        bet_mask = nib.load(bet_mask_path).get_fdata()

        # 마스크 바깥 영역 값 확인
        outside_brain = norm_img[bet_mask == 0]

        if np.any(outside_brain > 0):
            print(f"❌ Non-zero background: {base}", flush=True)
            unique_vals = np.unique(outside_brain[outside_brain > 0])
            print(f"   Unique vals: {unique_vals}", flush=True)
            nonzero_background_samples.append((base, unique_vals))

# 요약 결과 저장
if nonzero_background_samples:
    with open("/Users/iujeong/0.local/8.result/log/nonzero_background_summary.txt", "w") as f:
        for base, vals in nonzero_background_samples:
            f.write(f"{base}: {vals}\n")
    print("\n⚠️ 일부 샘플에서 배경 0 아님. 'nonzero_background_summary.txt' 확인", flush=True)
else:
    print("\n✅ 모든 배경이 0으로 정상입니다.", flush=True)

import csv

# 전체 배경 확인 결과 저장용 CSV
with open("/Users/iujeong/0.local/8.result/csv/nonzero_background_full_report.csv", "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["Patient", "Status", "Non-zero Background Values"])

    for root in root_dirs:
        norm_files = sorted(glob(os.path.join(root, "**", "*_norm.nii.gz"), recursive=True))
        print(f"[{root}]에서 {len(norm_files)}개 찾음", flush=True)
        for nf in norm_files:
            print(f" - {nf}", flush=True)
        for norm_path in tqdm(norm_files):
            print(f"Checking: {os.path.basename(norm_path)}", flush=True)
            base = os.path.basename(norm_path).replace("_norm.nii.gz", "")
            bet_mask_path = norm_path.replace("_norm.nii.gz", "_bet_mask.nii.gz")
            
            print(f" -> Looking for BET mask at: {bet_mask_path}", flush=True)
            if not os.path.exists(bet_mask_path):
                writer.writerow([base, "❌ BET mask not found", ""])
                continue

            norm_img = nib.load(norm_path).get_fdata()
            bet_mask = nib.load(bet_mask_path).get_fdata()

            outside_brain = norm_img[bet_mask == 0]

            if np.isnan(outside_brain).any():
                writer.writerow([base, "❌ NaN in background", "NaN"])
            elif np.any(outside_brain > 0):
                nonzero_vals = np.unique(outside_brain[outside_brain > 0])
                writer.writerow([base, "❌ Positive background", nonzero_vals])
            else:
                writer.writerow([base, "✅ OK", "≤ 0 only"])
