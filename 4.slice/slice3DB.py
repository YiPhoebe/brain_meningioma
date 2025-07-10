
import os
import numpy as np
import nibabel as nib
import pandas as pd
from scipy.stats import zscore
import imageio
from scipy.ndimage import center_of_mass


# 로그 루트 경로 상수
LOG_ROOT = "/Users/iujeong/0.local/8.result/log"



def get_mask_center(mask: np.ndarray):
    """
    3D 마스크(bet_mask, gtv_mask)에서 bounding box 중심 좌표를 정수형으로 반환한다.
    반환 값은 (cz, cy, cx) 순서인데
    cz는 굳이 쓰지 않지만 그래도 마스크가 3D라서 만들어 놓음
    """
    coords = np.argwhere(mask > 0)
    if coords.size == 0:
        return None
    z_min, y_min, x_min = coords.min(axis=0)
    z_max, y_max, x_max = coords.max(axis=0)
    cz = (z_min + z_max) // 2
    cy = (y_min + y_max) // 2
    cx = (x_min + x_max) // 2
    return cz, cy, cx  # 순서 주의!



def extract_numeric_id(pid: str) -> int:
    """
    환자 ID 문자열에서 숫자 ID만 뽑아내는 함수
    """
    return int(pid.split("-")[3])



def load_gtv_mask(patient_id: str, gtv_dir: str) -> np.ndarray:
    """
    주어진 환자 ID에 대해 GTV 마스크를 불러온다.
    gtv_dir에는 GTV 파일들이 patient_id_gtv_mask.nii.gz 형태로 저장되어 있어야 함
    """
    mask_path = os.path.join(gtv_dir, f"{patient_id}_gtv_mask.nii.gz")
    if not os.path.exists(mask_path):
        raise FileNotFoundError(f"GTV 마스크 파일 없음: {mask_path}")
    return nib.load(mask_path).get_fdata()



def load_bet_mask(patient_id: str, bet_dir: str) -> np.ndarray:
    """
    주어진 환자 ID에 대해 BET 마스크를 불러온다.
    bet_dir에는 patient_id_bet_mask.nii.gz 형태로 저장되어 있어야 함
    """
    mask_path = os.path.join(bet_dir, f"{patient_id}_bet_mask.nii.gz")
    if not os.path.exists(mask_path):
        raise FileNotFoundError(f"BET 마스크 파일 없음: {mask_path}")
    return nib.load(mask_path).get_fdata()



def filter_slices_by_mask_area(masks: np.ndarray, area_thresh: int = 10):
    """
    각 슬라이스별 마스크의 픽셀 수를 기반으로 필터링을 수행한다.

    ⚠️ 단 한 번만 수행하는 중요 필터링이므로 다음 기준에 따라 신중히 수행:
    
    1. 픽셀 수 너무 작은 슬라이스 제거:
        - np.sum(mask) < area_thresh 기준으로 제거
        - 뇌수막종의 특성상 일부 슬라이스에 거의 마스크가 없을 수 있으므로, 최소 기준만 적용

    """
    assert masks.ndim == 3  # (H, W, D)
    d = masks.shape[2]
    
    # 슬라이스별 마스크 픽셀 수
    areas = np.array([np.sum(masks[:, :, i]) for i in range(d)])

    # 1단계: 픽셀 수 < area_thresh 제거
    valid_idx = np.where(areas >= area_thresh)[0]

    # 2단계: 
    final_idx = valid_idx

    print(f"📊 슬라이스 필터링 결과 - 전체: {d}, 유지됨: {len(final_idx)}, 제거됨: {d - len(final_idx)}")

    return final_idx  # 남길 슬라이스 인덱스 리스트




 # 수정: BET 마스크 적용 버전 저장
def save_filtered_slices(
    volume: np.ndarray,
    mask: np.ndarray,
    keep_idx: list,
    out_npy_dir: str,
    out_png_dir: str,
    pid: str,
    bet: np.ndarray = None,
    # log_file: str = None
    log_file: str = None,
    bbox_coords: tuple = None
):
    """
    BET 마스크의 각 슬라이스에서 bounding box로 crop,
    crop 결과가 target_shape보다 작으면 중심 기준 padding, 크면 resize
    bbox center는 무시 (BET만 사용)
    """
    os.makedirs(out_npy_dir, exist_ok=True)
    os.makedirs(out_png_dir, exist_ok=True)
    

    coords = np.argwhere(bet > 0)
    if coords.size > 0:
        x_min, y_min, _ = coords.min(axis=0)
        x_max, y_max, _ = coords.max(axis=0) + 1
    else:
        raise ValueError(f"{pid}: BET 마스크 영역 내에 이미지가 없음 (비정상)")

    # --- 정규화 처리 (Normalization) ---
    # BET 마스크가 있으면 뇌 영역만 intensity 통계 산출, 없으면 전체 이미지 사용
    if bet is not None:
        brain_voxels = volume[bet > 0]  # BET 마스크 기준, 뇌 영역의 intensity만 사용
    else:
        brain_voxels = volume  # BET 없으면 전체 이미지 사용

    # BET 마스크 영역 내에 이미지가 없으면 예외 발생
    if brain_voxels.size == 0:
        raise ValueError(f"{pid}: BET 마스크 영역 내에 이미지가 없음 (비정상)")

    # 평균 intensity 계산
    global_mean = brain_voxels.mean()
    # 표준편차 계산
    global_std = brain_voxels.std()

    # 정규화 통계값을 CSV로 저장 (append mode)
    stat_row = pd.DataFrame([{
        "patient_id": pid,
        "mean": global_mean,
        "std": global_std
    }])
    stat_csv_path = os.path.join("/Users/iujeong/0.local/8.result/csv", "norm_stats.csv")
    os.makedirs(os.path.dirname(stat_csv_path), exist_ok=True)
    if not os.path.exists(stat_csv_path):
        stat_row.to_csv(stat_csv_path, index=False)
    else:
        stat_row.to_csv(stat_csv_path, mode='a', header=False, index=False)




    target_h, target_w = 160, 192  # 패딩 및 크롭 후 저장할 타겟 높이, 너비
    target_shape = (target_h, target_w)

    # --- BET 마스크를 이용해 마스크를 클리핑 ---
    # BET 마스크가 있으면, 마스크에서 뇌 외부 영역을 제거 (BET > 0인 영역만 남김)
    if bet is not None:
        mask = mask * (bet > 0)


    if log_file is not None:
        with open(log_file, "a") as f:
            f.write(f"3D xmin xmax ymin ymax-->: {x_min}, {x_max}, {y_min}, {y_max} \n")

    H, W = volume.shape[0:2]
    bbox_h = x_max - x_min
    bbox_w = y_max - y_min
    pad_h = (target_h - bbox_h) // 2
    pad_w = (target_w - bbox_w) // 2
    x_min_crop = max(x_min - pad_h,0)
    x_max_crop = min(x_min_crop + target_h,H)
    x_min_crop = x_max_crop - target_h
    y_min_crop = max(y_min - pad_w,0)
    y_max_crop = min(y_min_crop + target_w,W)
    y_min_crop = y_max_crop - target_w
    # 슬라이스 인덱스(keep_idx)에 대해 반복 처리
    for i in keep_idx:
        vol_slice = volume[:, :, i]  # i번째 슬라이스 추출 (이미지)
        mask_slice = mask[:, :, i]   # i번째 슬라이스 추출 (마스크)

        # 새로운 BET 마스크 기반 per-slice bounding box crop 로직
        if bet is not None:

                vol_slice = vol_slice[x_min_crop:x_max_crop, y_min_crop:y_max_crop]
                mask_slice = mask_slice[x_min_crop:x_max_crop, y_min_crop:y_max_crop]
                # bet_slice = bet_slice[x_min_crop:x_max_crop, y_min_crop:y_max_crop]
            # else:
            #     bet_slice = None
        # else:
        #     bet_slice = None


        # 타겟 크기(160, 192)로 패딩으로 크기 조정
        if (vol_slice.shape[0] < target_h or vol_slice.shape[1] < target_w):
            print("작은 사이즈--> 패딩")
            if log_file is not None:
                with open(log_file, "a") as f:
                    f.write("작은 사이즈--> 패딩\n")
            # vol_slice = padding(vol_slice, target_shape, bet_slice=bet_slice)
            # mask_slice = padding(mask_slice, target_shape, bet_slice=bet_slice)
        
        if (vol_slice.shape != (target_h, target_w)):
            print ('Error-->: shape 오류')
            if log_file is not None:
                with open(log_file, "a") as f:
                    f.write("Error-->: shape 오류\n")

        print(f"{pid} - cropped shape: {vol_slice.shape}")

        if log_file is not None:
            with open(log_file, "a") as f:
                f.write(f"{pid} - cropped shape: {vol_slice.shape}\n")

        # dtype 축소 (float32, uint8로)
        vol_slice = vol_slice.astype(np.float32)
        mask_slice = mask_slice.astype(np.uint8)
        
        # 슬라이스 저장 (npy 형식)
        np.save(os.path.join(out_npy_dir, f"{pid}_slice_{i:03d}_img.npy"), vol_slice)
        np.save(os.path.join(out_npy_dir, f"{pid}_slice_{i:03d}_mask.npy"), mask_slice)

    return "bet"


# 정규화한 영상 파일(그냥 norm 파일, mask 아님) 로딩
def load_image_volume(patient_id: str, img_dir: str) -> np.ndarray:
    # 환자 ID에 해당하는 정규화된 이미지 경로 생성
    img_path = os.path.join(img_dir, f"{patient_id}_norm.nii.gz")
    # 파일 존재 여부 확인
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"이미지 파일 없음: {img_path}")
    # NIfTI 파일을 불러와 numpy 배열로 반환
    return nib.load(img_path).get_fdata()


# patient_id 추출 함수
def extract_patient_id(filename: str) -> str:
    # 파일 이름에서 접미사 제거하여 순수한 환자 ID만 추출
    return (filename
        .replace("_norm.nii.gz", "")
        .replace("_gtv_mask.nii.gz", "")
        .replace("_bet_mask.nii.gz", "")
        .replace("_t1c.nii.gz", "")
        .replace(".nii.gz", ""))




# 👇 test/train/val 자동 분기
if __name__ == "__main__":
    input_base = "/Users/iujeong/0.local/3.normalize"
    out_base = "/Users/iujeong/0.local/4.slice"
    csv_dir = "/Users/iujeong/0.local/8.result/csv"

    test_ids = sorted([
        extract_patient_id(f) for f in os.listdir(os.path.join(input_base, "n_test"))
        if f.endswith("_norm.nii.gz")
    ], key=extract_numeric_id)
    train_ids = sorted([
        extract_patient_id(f) for f in os.listdir(os.path.join(input_base, "n_train"))
        if f.endswith("_norm.nii.gz")
    ], key=extract_numeric_id)
    val_ids = sorted([
        extract_patient_id(f) for f in os.listdir(os.path.join(input_base, "n_val"))
        if f.endswith("_norm.nii.gz")
    ], key=extract_numeric_id)

    csv_path = os.path.join(csv_dir, "bbox_stats.csv")

    for group, ids in [("test", test_ids), ("train", train_ids), ("val", val_ids)]:
        # 그룹별 예기치 않은 에러 케이스 리스트 초기화
        unexpected_error_cases = []
        # Initialize location log for centroid recording
        location_log = []
        img_dir = os.path.join(input_base, f"n_{group}")
        npy_out = os.path.join(out_base, f"s_{group}/npy")
        png_out = os.path.join(out_base, f"s_{group}/png")
        csv_path = csv_path if group == "test" else (csv_path if group == "train" else csv_path)
        exclude = []  # bbox_stats.csv에는 제외 기준 없음
        print(f"[{group.upper()}] 환자 수: {len(ids)} | 제외 대상 없음")
        gtv_base = os.path.join(input_base, f"n_{group}")

        # Read bet_bbox_stats.csv for bbox cropping
        bbox_df = pd.read_csv(os.path.join(csv_dir, "bet_bbox_stats.csv"))

        for pid in ids:
            print(f"[{group.upper()}] pid: {pid}, img_dir: {img_dir}, gtv_base: {gtv_base}")
            log_file = os.path.join(LOG_ROOT, f"filtered_{group}_3DB.log")
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
                gtv_ones = np.argwhere(gtv > 0)
                print(f"{pid} - gtv shape: {gtv_ones.min(axis=0), gtv_ones.max(axis=0)}, unique: {np.unique(gtv)}")
                with open(log_file, "a") as f:
                    f.write(f"{pid} - gtv shape: {gtv_ones.min(axis=0), gtv_ones.max(axis=0)}, unique: {np.unique(gtv)}\n")
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
                keep_idx = filter_slices_by_mask_area(gtv * (bet>0))
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
                        # 비연속적이면 별도 로그에도 기록
                        with open(os.path.join(LOG_ROOT, "non_contiguous_slices.log"), "a") as f:
                            f.write(f"{pid}\n")
                            f.flush()
                # Extract bbox_coords from bet_bbox_stats.csv
                row = bbox_df[bbox_df["patient_id"] == pid]
                if len(row) == 1:
                    x_min = int(row["x_min"].values[0])
                    x_max = int(row["x_max"].values[0])
                    y_min = int(row["y_min"].values[0])
                    y_max = int(row["y_max"].values[0])
                    bbox_coords = (x_min, x_max, y_min, y_max)
                else:
                    bbox_coords = None

                center_type = save_filtered_slices(img, gtv, keep_idx, npy_out, png_out, pid, bet=bet, log_file=log_file, bbox_coords=bbox_coords)

                print(f"{pid}: saved {len(keep_idx)} slices (BET mask loaded)")

                with open(log_file, "a") as f:
                    center_label = "bbox 중심 기준 crop" if center_type == "bbox" else "BET 중심 기준 crop"
                    f.write(f"{pid}: {len(keep_idx)}개 슬라이스 저장 완료 (✅ {center_label})\n")
                    f.flush()
                # After saving, check that slice files were saved
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
                # norm_log가 비어있으므로 해당 로그 저장 생략
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

        # Save GTV centroid locations for this group
        df_loc = pd.DataFrame(location_log)
        df_loc.to_csv(f"/Users/iujeong/0.local/8.result/csv/gtv_location_stats_{group}.csv", index=False)

        # Print unexpected errors for the current group if any
        if unexpected_error_cases:
            print("\n❌ 예기치 않게 제외된 케이스 목록:")
            for pid, err in unexpected_error_cases:
                print(f"- {pid}: {err}")

    # 디버깅 요약 정보 출력
    print("\n==== 디버깅 요약 ====")
    for group in ["train", "test", "val"]:
        log_path = os.path.join(LOG_ROOT, f"filtered_{group}_3DB.log")
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



