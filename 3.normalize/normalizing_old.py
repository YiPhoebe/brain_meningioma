# ========================================================
# ✅ 모델 학습용 슬라이스 저장 (Z-score 정규화 포함, 크기 보정 X)
# - HD-BET으로 스컬 제거된 T1c 이미지 기준
# - intensity normalization (Z-score, brain 영역 기준) 적용
# - 크기 보정은 적용하지 않음 (모델 입력 시 처리 예정)
# ========================================================
import nibabel as nib
import numpy as np
import os
import pandas as pd
from glob import glob


bet_dirs = [
    "/Users/iujeong/0.local/2.resample/r_test",
    "/Users/iujeong/0.local/2.resample/r_train",
    "/Users/iujeong/0.local/2.resample/r_val",
]

all_gtv_stats = []
all_bbox_stats = []

all_cases = []
for d in bet_dirs:
    phase = d.split("/")[-1].replace("r_", "")  # extract phase from path
    for f in sorted(glob(os.path.join(d, "*_t1c_gtv_mask.nii.gz"))):
        pid = os.path.basename(f).replace("_t1c_gtv_mask.nii.gz", "")
        all_cases.append((pid, d, phase))

for pid, bet_dir, phase in all_cases:
    gtv_clipping_stats = []
    bbox_stats = []

    # 각 슬라이스의 크기 기록 리스트
    slice_shape_stats = []

    img_path = os.path.join(bet_dir, f"{pid}_t1c_bet.nii.gz")
    mask_path = os.path.join(bet_dir, f"{pid}_t1c_gtv_mask.nii.gz")
    bet_mask_path = os.path.join(bet_dir, f"{pid}_t1c_bet_mask.nii.gz")

    if not os.path.exists(img_path) or not os.path.exists(mask_path) or not os.path.exists(bet_mask_path):
        print(f"{pid}: 필요한 파일 없음")
        continue

    img = nib.load(img_path).get_fdata()
    bet_mask = nib.load(bet_mask_path).get_fdata()

    # BET 마스크가 비었는지 검사
    if np.sum(bet_mask) == 0:
        print(f"{pid}: BET 마스크가 비어 있음 (skip)")
        # 로그 저장
        os.makedirs("/Users/iujeong/0.local/result/log", exist_ok=True)
        with open("/Users/iujeong/0.local/result/log/skipped_cases.txt", "a") as log_file:
            log_file.write(f"{pid}: BET 마스크가 비어 있음\n")
        continue

    # [추가] 전체 brain bounding box 계산
    x_any, y_any = np.any(bet_mask, axis=(1, 2)), np.any(bet_mask, axis=(0, 2))
    x_min, x_max = np.where(x_any)[0][[0, -1]]
    y_min, y_max = np.where(y_any)[0][[0, -1]]

    # margin 추가
    margin = 5
    x_min = max(0, x_min - margin)
    x_max = min(bet_mask.shape[0], x_max + margin)
    y_min = max(0, y_min - margin)
    y_max = min(bet_mask.shape[1], y_max + margin)

    # bounding box 정보 저장
    bbox_stats.append({
        "patient_id": pid,
        "phase": phase,
        "x_min": x_min, "x_max": x_max,
        "y_min": y_min, "y_max": y_max
    })

    gtv_mask = nib.load(mask_path).get_fdata()

    # GTV 바깥 제거
    original_gtv_voxels = np.sum(gtv_mask > 0)
    gtv_mask = gtv_mask * (bet_mask > 0)
    kept_gtv_voxels = np.sum(gtv_mask > 0)
    removed_ratio = 1 - (kept_gtv_voxels / original_gtv_voxels) if original_gtv_voxels > 0 else 0
    gtv_clipping_stats.append({
        "patient_id": pid,
        "phase": phase,
        "original": int(original_gtv_voxels),
        "kept": int(kept_gtv_voxels),
        "removed_ratio": removed_ratio
    })

    # === [1] Intensity Normalization (Z-score, brain 영역 기준) ===
    brain_pixels = img[bet_mask > 0]
    mean, std = brain_pixels.mean(), brain_pixels.std()
    img = (img - mean) / (std + 1e-8)
    # 정규화된 볼륨 저장
    normalized_save_dir = f"/Users/iujeong/0.local/3.normalize/norm_{phase}"
    os.makedirs(normalized_save_dir, exist_ok=True)
    nii = nib.load(img_path)
    nib.save(nib.Nifti1Image(img, affine=nii.affine), os.path.join(normalized_save_dir, f"{pid}_norm.nii.gz"))

    # 슬라이스 분할은 다른 단계에서 수행

    print(f"{pid}: 저장 완료")

    all_gtv_stats.extend(gtv_clipping_stats)
    all_bbox_stats.extend(bbox_stats)

# GTV 클리핑 통계 저장
os.makedirs("/Users/iujeong/0.local/result/log", exist_ok=True)
df = pd.DataFrame(all_gtv_stats)
df.to_csv("/Users/iujeong/0.local/result/csv/gtv_clipping_stats.csv", index=False)

# Bounding Box 저장
df_bbox = pd.DataFrame(all_bbox_stats)
df_bbox.to_csv("/Users/iujeong/0.local/result/csv/bbox_stats.csv", index=False)
