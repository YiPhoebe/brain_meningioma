
import os
import numpy as np
import nibabel as nib
import pandas as pd
from scipy.stats import zscore
import imageio
from scipy.ndimage import center_of_mass
import tqdm
import re

# ============================== #
#         Utility Functions      #
# ============================== #

def extract_numeric_id(pid: str) -> int:
    nums = re.findall(r'\d+', pid)
    return int(''.join(nums)) if nums else 0

def pad_to_shape(arr, target_shape=(160, 192), mode="constant"):
    h, w = arr.shape
    pad_h = max(target_shape[0] - h, 0)
    pad_w = max(target_shape[1] - w, 0)

    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left

    final = np.pad(arr, ((pad_top, pad_bottom), (pad_left, pad_right)), mode=mode)

    # Crop if necessary
    h_final, w_final = final.shape
    start_h = max((h_final - target_shape[0]) // 2, 0)
    start_w = max((w_final - target_shape[1]) // 2, 0)
    return final[start_h:start_h+target_shape[0], start_w:start_w+target_shape[1]]

def extract_patient_id(filename: str) -> str:
    return (filename
        .replace("_norm.nii.gz", "")
        .replace("_gtv_mask.nii.gz", "")
        .replace("_bet_mask.nii.gz", "")
        .replace("_t1c.nii.gz", "")
        .replace(".nii.gz", ""))

def filter_slices_by_mask_area(masks: np.ndarray, area_thresh: int = 10):
    """
    각 슬라이스별 마스크의 픽셀 수를 기반으로 필터링을 수행한다.
    """
    assert masks.ndim == 3  # (H, W, D)
    d = masks.shape[2]
    areas = np.array([np.sum(masks[:, :, i]) for i in range(d)])
    valid_idx = np.where(areas >= area_thresh)[0]
    final_idx = valid_idx
    print(f"📊 슬라이스 필터링 결과 - 전체: {d}, 유지됨: {len(final_idx)}, 제거됨: {d - len(final_idx)}")
    return final_idx

def load_image_volume(patient_id: str, base_dir: str) -> np.ndarray:
    img_path = os.path.join(base_dir, f"{patient_id}_norm.nii.gz")
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"이미지 파일 없음: {img_path}")
    return nib.load(img_path).get_fdata()

def load_gtv_mask(pid: str, base_dir: str) -> np.ndarray:
    path = os.path.join(base_dir, f"{pid}_gtv_mask.nii.gz")
    if not os.path.exists(path):
        raise FileNotFoundError(f"GTV 마스크 없음: {path}")
    return nib.load(path).get_fdata()

def load_bet_mask(pid: str, base_dir: str) -> np.ndarray:
    path = os.path.join(base_dir, f"{pid}_bet_mask.nii.gz")
    if not os.path.exists(path):
        raise FileNotFoundError(f"BET 마스크 없음: {path}")
    return nib.load(path).get_fdata()



# ====== Logical ordering of function definitions ====== #

def pad_to_shape(arr, target_shape=(160, 192), mode="constant"):
    h, w = arr.shape
    pad_h = max(target_shape[0] - h, 0)
    pad_w = max(target_shape[1] - w, 0)

    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left

    final = np.pad(arr, ((pad_top, pad_bottom), (pad_left, pad_right)), mode=mode)

    # Crop if necessary
    h_final, w_final = final.shape
    start_h = max((h_final - target_shape[0]) // 2, 0)
    start_w = max((w_final - target_shape[1]) // 2, 0)
    return final[start_h:start_h+target_shape[0], start_w:start_w+target_shape[1]]

def crop_by_bbox(img, x_min, x_max, y_min, y_max):
    """
    Crop a 2D image or mask given bounding box coordinates.
    """
    return img[x_min:x_max, y_min:y_max]

def save_filtered_slices(img, gtv, keep_idx, npy_out, png_out, pid, bet=None, log_file=None):
    """
    Save filtered slices as .npy files.
    """
    os.makedirs(npy_out, exist_ok=True)
    center_type = "bet"
    for idx in keep_idx:
        img_slice = pad_to_shape(img[:, :, idx], target_shape=(160, 192))
        mask_slice = pad_to_shape(gtv[:, :, idx], target_shape=(160, 192))
        # Save .npy files
        np.save(os.path.join(npy_out, f"{pid}_slice{idx:03d}_img.npy"), img_slice)
        np.save(os.path.join(npy_out, f"{pid}_slice{idx:03d}_mask.npy"), mask_slice)
    return center_type

def save_debug_crop(debug_crop_dir, patient_id, slice_idx, img_crop, mask_crop):
    """
    Save debug PNGs for cropped image and mask.
    """
    os.makedirs(debug_crop_dir, exist_ok=True)
    img_norm = (img_crop / (img_crop.max() + 1e-8) * 255).astype(np.uint8)
    mask_norm = (mask_crop * 255).astype(np.uint8)
    imageio.imwrite(
        os.path.join(debug_crop_dir, f"{patient_id}_slice{slice_idx:03d}_img.png"),
        img_norm
    )
    imageio.imwrite(
        os.path.join(debug_crop_dir, f"{patient_id}_slice{slice_idx:03d}_mask.png"),
        mask_norm
    )

def process_and_save_slices(input_dir, output_dir, bbox_df, target_shape=(160, 192), mode="constant"):
    """
    Processes and saves padded/cropped slices from input_dir to output_dir using bbox_df.
    """
    img_files = sorted([f for f in os.listdir(input_dir) if f.endswith("_img.npy")])
    for img_fname in tqdm.tqdm(img_files):
        base_name = img_fname.replace("_img.npy", "")
        mask_fname = f"{base_name}_mask.npy"

        img_path = os.path.join(input_dir, img_fname)
        mask_path = os.path.join(input_dir, mask_fname)

        if not os.path.exists(mask_path):
            print(f"⚠️ Missing mask for {img_fname}")
            continue

        img = np.load(img_path)
        mask = np.load(mask_path)

        if img.shape != mask.shape:
            print(f"❌ Shape mismatch: {img_fname}")
            print(f"   img shape: {img.shape}, mask shape: {mask.shape}")
            continue

        print(f"🔍 Processing: {img_fname}")
        print(f"   Original shape: {img.shape}")

        if bbox_df is None:
            print("❌ bbox_df is required for bbox-based cropping.")
            return

        row = bbox_df[bbox_df["slice_name"] == base_name]
        if row.empty:
            print(f"⚠️ No bbox info for {base_name}")
            continue

        # Check if image is large enough for cropping to target_shape
        if img.shape[0] < target_shape[0] or img.shape[1] < target_shape[1]:
            print(f"❌ {img_fname} skipped: shape {img.shape} smaller than target {target_shape}")
            continue

        bbox = row.iloc[0]
        x_min = int(bbox["x_min"])
        x_max = int(bbox["x_max"])
        y_min = int(bbox["y_min"])
        y_max = int(bbox["y_max"])

        # Check bbox bounds
        if x_max > img.shape[0] or y_max > img.shape[1]:
            print(f"❌ {img_fname} skipped: bbox ({x_min},{x_max},{y_min},{y_max}) exceeds image shape {img.shape}")
            continue

        print(f"   Crop coords: x=({x_min}, {x_max}), y=({y_min}, {y_max})")
        print(f"   Crop shape: {img[x_min:x_max, y_min:y_max].shape}")

        img_crop = crop_by_bbox(img, x_min, x_max, y_min, y_max)
        mask_crop = crop_by_bbox(mask, x_min, x_max, y_min, y_max)

        m = re.match(r"(.+)_slice(\d+)", base_name)
        if m:
            patient_id = m.group(1)
            slice_idx = int(m.group(2))
        else:
            patient_id = base_name
            slice_idx = 0

        print(f"   Final cropped shape: {img_crop.shape}")
        img_out_fname = f"{patient_id}_slice{slice_idx:03d}_img.npy"
        mask_out_fname = f"{patient_id}_slice{slice_idx:03d}_mask.npy"
        np.save(os.path.join(output_dir, img_out_fname), img_crop)
        np.save(os.path.join(output_dir, mask_out_fname), mask_crop)

# ============================== #
#         Main Execution         #
# ============================== #

if __name__ == "__main__":
    # ----------- Constants ----------- #
    LOG_ROOT = "/Users/iujeong/0.local/8.result/log"
    normalize_base = "/Users/iujeong/0.local/3.normalize"
    out_base = "/Users/iujeong/0.local/4.slice"
    csv_dir = "/Users/iujeong/0.local/8.result/csv"
    bbox_csv_path = os.path.join(csv_dir, "bet_bbox_stats.csv")
    bbox_df = pd.read_csv(bbox_csv_path)
    # --------------------------------- #

    group_dirs = {
        "test": os.path.join(normalize_base, "n_test"),
        "train": os.path.join(normalize_base, "n_train"),
        "val": os.path.join(normalize_base, "n_val"),
    }

    def list_patient_ids(group_dir):
        return sorted([
            extract_patient_id(f)
            for f in os.listdir(group_dir)
            if f.endswith("_norm.nii.gz")
        ], key=extract_numeric_id)

    test_ids = list_patient_ids(group_dirs["test"])
    train_ids = list_patient_ids(group_dirs["train"])
    val_ids = list_patient_ids(group_dirs["val"])

    csv_path = os.path.join(csv_dir, "bet_bbox_stats.csv")

    for group, ids in [("test", test_ids), ("train", train_ids), ("val", val_ids)]:
        # 그룹별 예기치 않은 에러 케이스 리스트 초기화
        unexpected_error_cases = []
        # Initialize location log for centroid recording
        location_log = []
        img_dir = group_dirs[group]
        npy_out = os.path.join(out_base, f"s_{group}/npy")
        png_out = os.path.join(out_base, f"s_{group}/png")
        csv_path = csv_path if group == "test" else (csv_path if group == "train" else csv_path)
        exclude = []  # bbox_stats.csv에는 제외 기준 없음
        print(f"[{group.upper()}] 환자 수: {len(ids)} | 제외 대상 없음")
        gtv_base = group_dirs[group]

        for pid in ids:
            print(f"[{group.upper()}] pid: {pid}, img_dir: {img_dir}, gtv_base: {gtv_base}")
            log_file = os.path.join(LOG_ROOT, f"filtered_{group}.log")
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            with open(log_file, "a") as f:
                f.write(f"[{group.upper()}] pid: {pid}, img_dir: {img_dir}, gtv_base: {gtv_base}\n")
                f.flush()
            if pid in exclude:
                print(f"제외된 환자: {pid}")
                with open(log_file, "a") as f:
                    f.write(f"{pid}: removed_ratio 기준으로 제외됨\n")
                    f.flush()
                continue
            try:
                gtv = load_gtv_mask(pid, gtv_base)
                print(f"{pid} - gtv shape: {gtv.shape}, unique: {np.unique(gtv)}")
                with open(log_file, "a") as f:
                    f.write(f"{pid} - gtv shape: {gtv.shape}, unique: {np.unique(gtv)}\n")
                    f.flush()
                bet = load_bet_mask(pid, gtv_base)  # BET 마스크도 불러오기

                # Shape mismatch check
                img = load_image_volume(pid, img_dir)
                if any(x.shape != img.shape for x in [gtv, bet]):
                    raise ValueError(f"{pid}: 이미지, GTV, BET shape이 다름 → img: {img.shape}, gtv: {gtv.shape}, bet: {bet.shape}")

                # Cast GTV and BET masks to uint8 binary
                gtv_bin = (gtv > 0.5).astype(np.uint8)
                bet_bin = (bet > 0.5).astype(np.uint8)

                # GTV가 BET 안에 전혀 포함되지 않으면 제외 (단 1 voxel도 없을 때만)
                intersection = ((gtv_bin > 0) & (bet_bin > 0)).sum()
                gtv_total = (gtv_bin > 0).sum()
                inside_ratio = intersection / (gtv_total + 1e-8)
                print(f"{pid} - intersection: {intersection}, gtv_total: {gtv_total}, inside_ratio: {inside_ratio:.6f}")
                with open(log_file, "a") as f:
                    f.write(f"{pid} - GTV와 BET 겹친 voxel 수: {intersection}, GTV 전체 voxel 수: {gtv_total}, 포함 비율: {inside_ratio:.6f}\n")
                    f.flush()
                if intersection == 0:
                    print(f"{pid}: GTV가 BET 영역 밖에 있어 제외됨")
                    with open(log_file, "a") as f:
                        f.write(f"{pid}: GTV가 BET 영역에 전혀 포함되지 않아 제외됨\n")
                        f.flush()
                    continue

                # 중심 좌표 기록용
                centroid = center_of_mass(gtv)
                if np.isnan(centroid).any():
                    print(f"{pid}: GTV center_of_mass 계산 실패 (NaN 포함)")
                    with open(log_file, "a") as f:
                        f.write(f"{pid}: GTV center_of_mass 계산 실패 (NaN 포함)\n")
                        f.flush()
                    continue
                location_log.append({"patient_id": pid, "x": centroid[0], "y": centroid[1], "z": centroid[2]})

                print(f"{pid} - img shape: {img.shape}, bet shape: {bet.shape}")
                with open(log_file, "a") as f:
                    f.write(f"{pid} - img shape: {img.shape}, bet shape: {bet.shape}\n")
                    f.flush()
                keep_idx = filter_slices_by_mask_area(gtv)
                if len(keep_idx) == 0:
                    print(f"⚠️  {pid} - 모든 슬라이스가 필터링됨 (keep_idx 비어있음)")
                    with open(log_file, "a") as f:
                        f.write(f"⚠️  {pid} - 모든 슬라이스가 필터링됨 (keep_idx 비어있음)\n")
                        f.flush()
                else:
                    print(f"✅ {pid} - 남은 슬라이스 개수: {len(keep_idx)} / 전체: {gtv.shape[2]}")
                    with open(log_file, "a") as f:
                        f.write(f"✅ {pid} - 남은 슬라이스 개수: {len(keep_idx)} / 전체: {gtv.shape[2]}\n")
                        f.flush()
                print(f"{pid} - keep_idx: {keep_idx}")
                with open(log_file, "a") as f:
                    f.write(f"{pid} - keep_idx: {keep_idx}\n")
                    f.flush()
                if len(keep_idx) == 0:
                    with open(log_file, "a") as f:
                        f.write(f"{pid}: 필터링 후 남은 슬라이스 없음\n")
                        f.flush()
                else:
                    # 저장된 슬라이스 인덱스가 연속적인지 확인
                    sorted_idx = sorted(keep_idx)
                    if any((sorted_idx[i+1] - sorted_idx[i]) != 1 for i in range(len(sorted_idx)-1)):
                        with open(log_file, "a") as f:
                            f.write(f"{pid}: ⚠️ 저장된 슬라이스 인덱스가 연속되지 않음\n")
                            f.flush()
                        with open(os.path.join(LOG_ROOT, "non_contiguous_slices.log"), "a") as f:
                            f.write(f"{pid}\n")
                            f.flush()
                center_type = save_filtered_slices(img, gtv, keep_idx, npy_out, png_out, pid, bet=bet, log_file=log_file)
                print(f"{pid}: saved {len(keep_idx)} slices (BET mask loaded)")
                with open(log_file, "a") as f:
                    center_label = "bbox 중심 기준 crop" if center_type == "bbox" else "BET 중심 기준 crop"
                    f.write(f"{pid}: {len(keep_idx)}개 슬라이스 저장 완료 (✅ {center_label})\n")
                    f.flush()
                saved_slices = len([f for f in os.listdir(npy_out) if f.startswith(pid) and f.endswith("_img.npy")])
                if saved_slices == 0:
                    print(f"{pid}: 저장된 슬라이스 없음 (파일 확인)")
                    with open(log_file, "a") as f:
                        f.write(f"{pid}: 저장된 슬라이스 없음 (파일 확인)\n")
                        f.flush()
                print(f"✅ 저장 완료: {pid}")
                with open(log_file, "a") as f:
                    f.write(f"✅ 저장 완료: {pid}\n")
                    f.flush()
            except FileNotFoundError as e:
                print(f"파일 없음 에러: {e}")
                continue
            except Exception as e:
                err_msg = str(e)
                if "No slices saved due to BET masking" in err_msg:
                    err_msg_kor = "BET 마스킹으로 인해 저장된 슬라이스 없음"
                elif "GTV is completely outside the BET region" in err_msg:
                    err_msg_kor = "GTV가 BET 영역 밖에 있음"
                elif "img.shape != gtv.shape" in err_msg:
                    err_msg_kor = "이미지와 GTV의 shape 불일치"
                elif "object of type 'float' has no len()" in err_msg:
                    err_msg_kor = "슬라이스 개수 계산 중 float 오류"
                elif "cannot convert float NaN to integer" in err_msg:
                    err_msg_kor = "NaN 값을 정수로 변환할 수 없음 (슬라이스 범위 계산 오류)"
                elif "could not broadcast input array" in err_msg:
                    err_msg_kor = "슬라이스 저장 시 shape 불일치 (배열 브로드캐스트 오류)"
                elif "index" in err_msg and "is out of bounds" in err_msg:
                    err_msg_kor = "슬라이스 인덱스가 범위를 벗어남"
                else:
                    err_msg_kor = err_msg
                print(f"{pid}: ❌ 예기치 않은 오류로 제외됨 → {err_msg_kor}")
                with open(log_file, "a") as f:
                    f.write(f"{pid}: ❌ 예기치 않은 오류 발생 → {err_msg_kor}\n")
                    f.flush()
                unexpected_error_cases.append((pid, err_msg_kor))

        df_loc = pd.DataFrame(location_log)
        df_loc.to_csv(f"/Users/iujeong/0.local/8.result/csv/gtv_location_stats_{group}.csv", index=False)
        if unexpected_error_cases:
            print("\n❌ 예기치 않게 제외된 케이스 목록:")
            for pid, err in unexpected_error_cases:
                print(f"- {pid}: {err}")

    print("\n==== 디버깅 요약 ====")
    for group in ["train", "test", "val"]:
        log_path = os.path.join(LOG_ROOT, f"filtered_{group}.log")
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

    train_set = set(train_ids)
    test_set = set(test_ids)
    overlap = train_set & test_set
    if overlap:
        print(f"\n⚠️ 경고: Train/Test/Val에 중복된 환자 존재: {len(overlap)}명")
        for pid in sorted(overlap):
            print(f" - {pid}")
    else:
        print("\n✅ Train/Test 환자 ID 완전히 분리됨")
