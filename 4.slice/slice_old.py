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
    mask_path = os.path.join(base_dir, f"{patient_id}_t1c_bet_mask.nii.gz")
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

    for i in keep_idx:
        vol_slice = volume[:, :, i]
        mask_slice = mask[:, :, i]

        if bet is not None:
            # BET 마스크는 곱하지 않고 좌표만 사용
            bet_bin = (bet[:, :, i] > 0.5).astype(np.uint8)
            coords = np.argwhere(bet_bin)
            if coords.size > 0:
                x_min, y_min = coords.min(axis=0)
                x_max, y_max = coords.max(axis=0) + 1
                vol_slice = vol_slice[x_min:x_max, y_min:y_max]
                mask_slice = mask_slice[x_min:x_max, y_min:y_max]

        # Save as .npy files
        np.save(os.path.join(out_npy_dir, f"{pid}_slice_{i:03d}_img.npy"), vol_slice)
        np.save(os.path.join(out_npy_dir, f"{pid}_slice_{i:03d}_mask.npy"), mask_slice)

        # Normalize vol_slice using global Z-score, then clip and rescale to [0, 1]
        z_norm = (vol_slice - global_mean) / (global_std + 1e-8)
        z_clipped = np.clip(z_norm, -2, 2)
        norm_slice = (z_clipped + 2) / 4  # scale to [0, 1]
        imageio.imwrite(os.path.join(out_png_dir, f"{pid}_slice_{i:03d}_img.png"), (norm_slice * 255).astype(np.uint8))
        imageio.imwrite(os.path.join(out_png_dir, f"{pid}_slice_{i:03d}_mask.png"), (mask_slice * 255).astype(np.uint8))

        if bet is not None:
            imageio.imwrite(os.path.join(out_png_dir, f"{pid}_slice_{i:03d}_bet.png"), (bet[:, :, i] * 255).astype(np.uint8))


# 새 함수: nom_test/nom_train에서 볼륨 불러오기
def load_image_volume(patient_id: str, base_dir: str) -> np.ndarray:
    img_path = os.path.join(base_dir, f"{patient_id}_norm.nii.gz")
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"이미지 파일 없음: {img_path}")
    return nib.load(img_path).get_fdata()

# patient_id 추출 함수
def extract_patient_id(filename: str) -> str:
    return filename.replace("_norm.nii.gz", "").replace("_t1c.nii.gz", "").replace(".nii.gz", "")

# 👇 test/train 자동 분기
if __name__ == "__main__":
    input_base = "/home/iujeong/brain_meningioma/prepocessing/normalized_volume"
    out_base = "/home/iujeong/brain_meningioma/slice/filter"
    csv_dir = "/home/iujeong/brain_meningioma/outputs/csv"
    bet_base_train = "/home/iujeong/brain_meningioma/bet_output/bet_output_train"
    bet_base_test = "/home/iujeong/brain_meningioma/bet_output/bet_output_test"

    test_ids = sorted([
        extract_patient_id(f) for f in os.listdir(os.path.join(input_base, "norm_test"))
    ])
    train_ids = sorted([
        extract_patient_id(f) for f in os.listdir(os.path.join(input_base, "norm_train"))
    ])
    valid_ids = sorted([
        extract_patient_id(f) for f in os.listdir(os.path.join(input_base, "norm_valid"))
    ])

    test_csv_path = os.path.join(csv_dir, "gtv_clipping_stats_test.csv")
    train_csv_path = os.path.join(csv_dir, "gtv_clipping_stats.csv")

    for group, ids in [("test", test_ids), ("train", train_ids), ("valid", valid_ids)]:
        location_log = []
        img_dir = os.path.join(input_base, f"norm_{group}")
        npy_out = os.path.join(out_base, f"slice_{group}_f/npy")
        png_out = os.path.join(out_base, f"slice_{group}_f/png")
        csv_path = test_csv_path if group == "test" else (train_csv_path if group == "train" else train_csv_path)
        exclude = filter_patient_by_csv(csv_path)
        gtv_base = bet_base_test if group == "test" else (bet_base_train if group == "train" else bet_base_test)

        for pid in ids:
            log_file = os.path.join("/home/iujeong/brain_meningioma/slice/filter/filter_log", f"filtered_{group}.log")
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            if pid in exclude:
                with open(log_file, "a") as f:
                    f.write(f"{pid}: excluded by removed_ratio threshold\n")
                continue
            try:
                gtv = load_gtv_mask(pid, gtv_base)
                print(f"{pid} - gtv shape: {gtv.shape}, unique: {np.unique(gtv)}")
                bet = load_bet_mask(pid, gtv_base)  # BET 마스크도 불러오기

                # 중심 좌표 기록용
                centroid = center_of_mass(gtv)
                location_log.append({"patient_id": pid, "x": centroid[0], "y": centroid[1], "z": centroid[2]})

                img = load_image_volume(pid, img_dir)
                print(f"{pid} - img shape: {img.shape}, bet shape: {bet.shape}")
                keep_idx = filter_slices_by_mask_area(gtv)
                if len(keep_idx) == 0:
                    print(f"⚠️  {pid} - 모든 슬라이스가 필터링됨 (keep_idx 비어있음)")
                else:
                    print(f"✅ {pid} - 남은 슬라이스 개수: {len(keep_idx)} / 전체: {gtv.shape[2]}")
                print(f"{pid} - keep_idx: {keep_idx}")
                if len(keep_idx) == 0:
                    with open(log_file, "a") as f:
                        f.write(f"{pid}: 0 slices remained after filtering\n")
                else:
                    # 저장된 슬라이스 인덱스가 연속적인지 확인
                    sorted_idx = sorted(keep_idx)
                    if any((sorted_idx[i+1] - sorted_idx[i]) != 1 for i in range(len(sorted_idx)-1)):
                        with open(log_file, "a") as f:
                            f.write(f"{pid}: warning - saved slice indices are not contiguous\n")
                save_filtered_slices(img, gtv, keep_idx, npy_out, png_out, pid, bet=bet)
                print(f"{pid}: saved {len(keep_idx)} slices (BET mask loaded)")
            except FileNotFoundError as e:
                # norm_log가 비어있으므로 해당 로그 저장 생략
                continue
            except Exception as e:
                print(f"Skip {pid} due to unexpected error: {e}")

        df_loc = pd.DataFrame(location_log)
        df_loc.to_csv(f"/home/iujeong/brain_meningioma/outputs/csv/gtv_location_stats_{group}.csv", index=False)

    # 디버깅 요약 정보 출력
    print("\n==== 디버깅 요약 ====")
    for group in ["train", "test"]:
        log_path = f"/home/iujeong/brain_meningioma/slice/filter/filter_log/filtered_{group}.log"
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
        print(f"\n⚠️ 경고: Train/Test에 중복된 환자 존재: {len(overlap)}명")
        for pid in sorted(overlap):
            print(f" - {pid}")
    else:
        print("\n✅ Train/Test 환자 ID 완전히 분리됨")
