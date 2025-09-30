import os
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm

# 데이터 루트 경로
root_dirs = [
    "/Users/iujeong/0.local/4.slice/s_train/npy",
    "/Users/iujeong/0.local/4.slice/s_val/npy",
    "/Users/iujeong/0.local/4.slice/s_test/npy"
]

# 결과 저장 리스트
records = []

for root in root_dirs:
    npy_files = sorted(glob(os.path.join(root, "*_mask.npy")))
    print(f"Searching in: {root}, Found {len(npy_files)} mask files.")
    for mask_path in tqdm(npy_files):
        try:
            mask = np.load(mask_path)
            print(f"Checking {mask_path} - sum: {np.sum(mask)}")
            if np.sum(mask) == 0:
                continue  # 종양 없음

            # 중심 좌표 계산
            tumor_coords = np.argwhere(mask)
            centroid = tumor_coords.mean(axis=0)  # (y, x)

            # Region 태깅 기준 (확장된 분류)
            x, y = centroid[1], centroid[0]
            if y < 50:
                region = "Frontal"
            elif 50 <= y < 90:
                if x < 80:
                    region = "Temporal_L"
                elif x > 120:
                    region = "Temporal_R"
                else:
                    region = "Parietal"
            elif 90 <= y < 130:
                if x < 70:
                    region = "Temporal_L"
                elif x > 130:
                    region = "Temporal_R"
                else:
                    region = "Parietal"
            elif y >= 130:
                if x < 90:
                    region = "Cerebellum_L"
                elif x > 110:
                    region = "Cerebellum_R"
                else:
                    region = "Occipital"
            else:
                region = "Other"

            # 환자 ID 및 slice index 추출
            fname = os.path.basename(mask_path)
            slice_part = fname.split("_slice_")[-1]
            patient_id = fname.replace(f"_slice_{slice_part}", "")
            slice_idx = slice_part.replace("_mask.npy", "")

            records.append({
                "PatientID": patient_id,
                "SliceIdx": int(slice_idx),
                "Y": float(centroid[0]),
                "X": float(centroid[1]),
                "Region": region,
                "MaskPath": mask_path
            })

            # 예측 마스크 경로 추정
            pred_path = mask_path.replace("_mask.npy", "_pred.npy")
            if os.path.exists(pred_path):
                pred = np.load(pred_path)
                gt = mask
                pred_bin = (pred > 0.5).astype(np.uint8)
                gt_bin = (gt > 0.5).astype(np.uint8)
                intersection = np.logical_and(pred_bin, gt_bin).sum()
                union = np.logical_or(pred_bin, gt_bin).sum()
                iou = intersection / union if union != 0 else 0.0
            else:
                iou = np.nan

            records[-1]["IoU"] = iou
            if os.path.exists(pred_path) and np.sum(pred_bin) > 0:
                pred_coords = np.argwhere(pred_bin)
                pred_centroid = pred_coords.mean(axis=0)
                records[-1]["PredY"] = float(pred_centroid[0])
                records[-1]["PredX"] = float(pred_centroid[1])
            else:
                records[-1]["PredY"] = np.nan
                records[-1]["PredX"] = np.nan
        except Exception as e:
            print(f"Error processing {mask_path}: {e}")

# 저장
df = pd.DataFrame(records)
csv_path = "/Users/iujeong/0.local/4.slice/tumor_centers.csv"
df.to_csv(csv_path, index=False)
print(f"Saved tumor center coordinates to : {csv_path}")