# Numeric patient ID extraction helper for sorting
def extract_numeric_id(pid: str) -> int:
    return int(pid.split("-")[3])
import os
import nibabel as nib
import pandas as pd
import numpy as np
from scipy.stats import zscore
import imageio
from scipy.ndimage import center_of_mass

def filter_patient_by_csv(csv_path: str, removed_threshold: float = 1.0):
    """
    CSV 파일에서 removed_ratio >= threshold 인 환자 목록을 반환
    """
    df = pd.read_csv(csv_path)
    return df[df["removed_ratio"] >= removed_threshold]["patient_id"].tolist()

def load_gtv_mask(patient_id: str, base_dir: str) -> np.ndarray:
    """
    주어진 환자 ID에 대해 GTV 마스크를 불러온다.
    base_dir에는 GTV 파일들이 patient_id_gtv_mask.nii.gz 형태로 저장되어 있어야 함
    """
    mask_path = os.path.join(base_dir, f"{patient_id}_gtv_mask.nii.gz")
    if not os.path.exists(mask_path):
        raise FileNotFoundError(f"GTV 마스크 파일 없음: {mask_path}")
    return nib.load(mask_path).get_fdata()

def load_bet_mask(patient_id: str, base_dir: str) -> np.ndarray:
    """
    주어진 환자 ID에 대해 BET 마스크를 불러온다.
    base_dir에는 patient_id_bet_mask.nii.gz 형태로 저장되어 있어야 함
    """
    mask_path = os.path.join(base_dir, f"{patient_id}_bet_mask.nii.gz")
    if not os.path.exists(mask_path):
        raise FileNotFoundError(f"BET 마스크 파일 없음: {mask_path}")
    return nib.load(mask_path).get_fdata()

def filter_slices_by_mask_area(gtv_mask, area_thresh=10, z_thresh=2.5):
    areas = np.sum(gtv_mask, axis=(1, 2))
    z_scores = zscore(areas)

    keep_slices = []
    for z, area, z_val in zip(range(len(areas)), areas, z_scores):
        if area >= area_thresh and abs(z_val) < z_thresh:
            keep_slices.append(z)
    return keep_slices


# 수정: BET 마스크 적용 버전 저장
def save_filtered_slices(volume: np.ndarray, mask: np.ndarray, keep_idx: list, out_npy_dir: str, out_png_dir: str, pid: str, bet: np.ndarray = None):
    os.makedirs(out_npy_dir, exist_ok=True)
    os.makedirs(out_png_dir, exist_ok=True)

    if bet is not None:
        brain_voxels = volume[bet > 0]
    else:
        brain_voxels = volume
    global_mean = brain_voxels.mean()
    global_std = brain_voxels.std()

    saved_count = 0

    for i in keep_idx:
        vol_slice = volume[:, :, i]
        mask_slice = mask[:, :, i]

        # 패딩 크기 정의
        PAD_H, PAD_W = 160, 192

        # 중심 기준 패딩
        if bet is not None:
            bet_slice = bet[:, :, i]
            if np.sum(bet_slice) == 0:
                print(f"⚠️ {pid} - BET slice {i} is empty, fallback to GTV center")
                cy, cx = center_of_mass(mask_slice)
            else:
                cy, cx = center_of_mass(bet_slice)
        else:
            cy, cx = center_of_mass(mask_slice)
        cy = int(round(cy))
        cx = int(round(cx))
        y_start = max(0, cy - PAD_H // 2)
        x_start = max(0, cx - PAD_W // 2)
        y_end = y_start + PAD_H
        x_end = x_start + PAD_W

        # 슬라이스 크기에 맞춰 잘라내고, 넘치면 패딩 추가
        vol_crop = np.zeros((PAD_H, PAD_W), dtype=vol_slice.dtype)
        mask_crop = np.zeros((PAD_H, PAD_W), dtype=mask_slice.dtype)

        y_slice = slice(y_start, min(y_end, vol_slice.shape[0]))
        x_slice = slice(x_start, min(x_end, vol_slice.shape[1]))
        y_offset = max(0, - (cy - PAD_H // 2))
        x_offset = max(0, - (cx - PAD_W // 2))

        vol_crop[y_offset:y_offset + (y_slice.stop - y_slice.start),
                 x_offset:x_offset + (x_slice.stop - x_slice.start)] = vol_slice[y_slice, x_slice]
        mask_crop[y_offset:y_offset + (y_slice.stop - y_slice.start),
                  x_offset:x_offset + (x_slice.stop - x_slice.start)] = mask_slice[y_slice, x_slice]

        vol_slice = vol_crop
        mask_slice = mask_crop

        # Save as .npy files
        np.save(os.path.join(out_npy_dir, f"{pid}_slice_{i:03d}_img.npy"), vol_slice)
        np.save(os.path.join(out_npy_dir, f"{pid}_slice_{i:03d}_mask.npy"), mask_slice)

        # Normalize vol_slice using global Z-score, then clip and rescale to [0, 1]
        z_norm = (vol_slice - global_mean) / (global_std + 1e-8)
        z_clipped = np.clip(z_norm, -2, 2)
        norm_slice = (z_clipped + 2) / 4  # scale to [0, 1]
        imageio.imwrite(os.path.join(out_png_dir, f"{pid}_slice_{i:03d}_img.png"), (norm_slice * 255).astype(np.uint8))
        # Removed saving mask PNG as per instructions

        saved_count += 1

    if saved_count == 0:
        raise ValueError("No slices saved due to BET masking")

    print(f"{pid}: 실제 저장된 슬라이스 개수: {saved_count}")


# 새 함수: nom_test/nom_train에서 볼륨 불러오기
def load_image_volume(patient_id: str, base_dir: str) -> np.ndarray:
    img_path = os.path.join(base_dir, f"{patient_id}_norm.nii.gz")
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"이미지 파일 없음: {img_path}")
    return nib.load(img_path).get_fdata()

# patient_id 추출 함수
def extract_patient_id(filename: str) -> str:
    return (filename
        .replace("_norm.nii.gz", "")
        .replace("_gtv_mask.nii.gz", "")
        .replace("_bet_mask.nii.gz", "")
        .replace("_t1c.nii.gz", "")
        .replace(".nii.gz", ""))

# 👇 test/train 자동 분기
if __name__ == "__main__":
    input_base = "/Users/iujeong/0.local/3.normalize"
    out_base = "/Users/iujeong/0.local/4.slice"
    csv_dir = "/Users/iujeong/0.local/8.result/csv"

    test_ids = sorted([
        extract_patient_id(f) for f in os.listdir(os.path.join(input_base, "n_test", "nii"))
        if f.endswith("_norm.nii.gz")
    ], key=extract_numeric_id)
    train_ids = sorted([
        extract_patient_id(f) for f in os.listdir(os.path.join(input_base, "n_train", "nii"))
        if f.endswith("_norm.nii.gz")
    ], key=extract_numeric_id)
    val_ids = sorted([
        extract_patient_id(f) for f in os.listdir(os.path.join(input_base, "n_val", "nii"))
        if f.endswith("_norm.nii.gz")
    ], key=extract_numeric_id)

    csv_path = os.path.join(csv_dir, "bbox_stats.csv")

    for group, ids in [("test", test_ids), ("train", train_ids), ("val", val_ids)]:
        location_log = []
        img_dir = os.path.join(input_base, f"n_{group}", "nii")
        npy_out = os.path.join(out_base, f"s_{group}/npy")
        png_out = os.path.join(out_base, f"s_{group}/png")
        csv_path = csv_path if group == "test" else (csv_path if group == "train" else csv_path)
        exclude = []  # bbox_stats.csv에는 제외 기준 없음
        print(f"[{group.upper()}] 환자 수: {len(ids)} | 제외 대상 없음")
        gtv_base = os.path.join(input_base, f"n_{group}", "nii")

        for pid in ids:
            print(f"[{group.upper()}] pid: {pid}, img_dir: {img_dir}, gtv_base: {gtv_base}")
            log_file = os.path.join("/Users/iujeong/0.local/8.result/log", f"filtered_{group}.log")
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            with open(log_file, "a") as f:
                f.write(f"[{group.upper()}] pid: {pid}, img_dir: {img_dir}, gtv_base: {gtv_base}\n")
            if pid in exclude:
                print(f"제외된 환자: {pid}")
                with open(log_file, "a") as f:
                    f.write(f"{pid}: removed_ratio 기준으로 제외됨\n")
                continue
            try:
                gtv = load_gtv_mask(pid, gtv_base)
                print(f"{pid} - gtv shape: {gtv.shape}, unique: {np.unique(gtv)}")
                with open(log_file, "a") as f:
                    f.write(f"{pid} - gtv shape: {gtv.shape}, unique: {np.unique(gtv)}\n")
                bet = load_bet_mask(pid, gtv_base)  # BET 마스크도 불러오기

                # GTV가 BET 안에 전혀 포함되지 않으면 제외 (단 1 voxel도 없을 때만)
                gtv_bin = (gtv > 0.5).astype(np.uint8)
                bet_bin = (bet > 0.5).astype(np.uint8)
                intersection = ((gtv_bin > 0) & (bet_bin > 0)).sum()
                gtv_total = (gtv_bin > 0).sum()
                inside_ratio = intersection / (gtv_total + 1e-8)
                print(f"{pid} - intersection: {intersection}, gtv_total: {gtv_total}, inside_ratio: {inside_ratio:.6f}")
                with open(log_file, "a") as f:
                    f.write(f"{pid} - intersection: {intersection}, gtv_total: {gtv_total}, inside_ratio: {inside_ratio:.6f}\n")
                if intersection == 0:
                    print(f"{pid}: GTV가 BET 영역 밖에 있어 제외됨")
                    with open(log_file, "a") as f:
                        f.write(f"{pid}: GTV가 BET 영역에 전혀 포함되지 않아 제외됨\n")
                    continue

                # 중심 좌표 기록용
                centroid = center_of_mass(gtv)
                location_log.append({"patient_id": pid, "x": centroid[0], "y": centroid[1], "z": centroid[2]})

                img = load_image_volume(pid, img_dir)
                print(f"{pid} - img shape: {img.shape}, bet shape: {bet.shape}")
                with open(log_file, "a") as f:
                    f.write(f"{pid} - img shape: {img.shape}, bet shape: {bet.shape}\n")
                keep_idx = filter_slices_by_mask_area(gtv)
                if len(keep_idx) == 0:
                    print(f"⚠️  {pid} - 모든 슬라이스가 필터링됨 (keep_idx 비어있음)")
                    with open(log_file, "a") as f:
                        f.write(f"⚠️  {pid} - 모든 슬라이스가 필터링됨 (keep_idx 비어있음)\n")
                else:
                    print(f"✅ {pid} - 남은 슬라이스 개수: {len(keep_idx)} / 전체: {gtv.shape[2]}")
                    with open(log_file, "a") as f:
                        f.write(f"✅ {pid} - 남은 슬라이스 개수: {len(keep_idx)} / 전체: {gtv.shape[2]}\n")
                print(f"{pid} - keep_idx: {keep_idx}")
                with open(log_file, "a") as f:
                    f.write(f"{pid} - keep_idx: {keep_idx}\n")
                if len(keep_idx) == 0:
                    with open(log_file, "a") as f:
                        f.write(f"{pid}: 필터링 후 남은 슬라이스 없음\n")
                else:
                    # 저장된 슬라이스 인덱스가 연속적인지 확인
                    sorted_idx = sorted(keep_idx)
                    if any((sorted_idx[i+1] - sorted_idx[i]) != 1 for i in range(len(sorted_idx)-1)):
                        with open(log_file, "a") as f:
                            f.write(f"{pid}: ⚠️ 저장된 슬라이스 인덱스가 연속되지 않음\n")
                save_filtered_slices(img, gtv, keep_idx, npy_out, png_out, pid, bet=bet)
                print(f"{pid}: saved {len(keep_idx)} slices (BET mask loaded)")
                with open(log_file, "a") as f:
                    f.write(f"{pid}: saved {len(keep_idx)} slices (BET mask loaded)\n")
                print(f"✅ 저장 완료: {pid}")
                with open(log_file, "a") as f:
                    f.write(f"✅ 저장 완료: {pid}\n")
            except FileNotFoundError as e:
                print(f"파일 없음 에러: {e}")
                # norm_log가 비어있으므로 해당 로그 저장 생략
                continue
            except Exception as e:
                if "No slices saved due to BET masking" in str(e):
                    print(f"{pid}: all slices skipped due to BET masking")
                    with open(log_file, "a") as f:
                        f.write(f"{pid}: BET 마스크로 인해 모든 슬라이스 제외됨\n")
                elif "GTV is completely outside the BET region" in str(e):
                    print(f"{pid}: GTV가 BET 영역 밖에 있어 제외됨")
                    with open(log_file, "a") as f:
                        f.write(f"{pid}: GTV가 BET 영역에 전혀 포함되지 않아 제외됨\n")
                else:
                    print(f"Skip {pid} due to unexpected error: {e}")
                    with open(log_file, "a") as f:
                        f.write(f"{pid}: ❌ 예기치 않은 오류로 제외됨 → {str(e)}\n")

        df_loc = pd.DataFrame(location_log)

    # 디버깅 요약 정보 출력
    print("\n==== 디버깅 요약 ====")
    for group in ["train", "test", "val"]:
        log_path = f"/Users/iujeong/0.local/8.result/log/filtered_{group}.log"
        total = 0
        saved = 0
        zero = 0
        if os.path.exists(log_path):
            with open(log_path, "r") as f:
                for line in f:
                    total += 1
                    if "0 slices remained" in line:
                        zero += 1
                    elif "saved" in line:
                        saved += 1
        print(f"[{group}] 전체 환자 수: {total}, 저장된 환자: {saved}, 제거된 환자: {zero}")

    # train/test ID 겹침 여부 확인
    train_set = set(train_ids)
    test_set = set(test_ids)
    overlap = train_set & test_set
    if overlap:
        print(f"\n⚠️ 경고: Train/Test/Val에 중복된 환자 존재: {len(overlap)}명")
        for pid in sorted(overlap):
            print(f" - {pid}")
    else:
        print("\n✅ Train/Test 환자 ID 완전히 분리됨")
