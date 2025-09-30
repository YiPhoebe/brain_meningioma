import os

# 환자 ID 추출 함수
def get_patient_ids_from_dir(directory, delimiter="_"):
    filenames = os.listdir(directory)
    patient_ids = set(f.split(delimiter)[0] for f in filenames if f.endswith(".nii.gz") or f.endswith(".npy"))
    return patient_ids

# 로그 경로
log_path = "/Users/iujeong/0.local/4.slice/check_code/slicing_dropped_cases.log"
with open(log_path, "w") as f:
    f.write("")

splits = ["train", "val", "test"]

for split in splits:
    print(f"\n===== {split.upper()} =====")
    normalized_dir = f"/Users/iujeong/0.local/3.normalize/n_{split}"
    sliced_dir = f"/Users/iujeong/0.local/4.slice/s_{split}/npy"
    resampled_dir = f"/Users/iujeong/0.local/2.resample/r_{split}"

    # ID 추출
    normalized_ids = get_patient_ids_from_dir(normalized_dir)
    sliced_ids = get_patient_ids_from_dir(sliced_dir)
    resampled_ids = get_patient_ids_from_dir(resampled_dir)

    # 슬라이싱 도중 누락
    dropped_ids = sorted(normalized_ids - sliced_ids)
    print(f"정규화까지 됐지만 슬라이싱 도중 제외된 환자 수: {len(dropped_ids)}명")
    for pid in dropped_ids:
        print(f"❌ Dropped at slicing stage: {pid}")

    # 정규화 도중 누락
    normalize_dropped_ids = sorted(resampled_ids - normalized_ids)
    print(f"정규화 도중 제외된 환자 수: {len(normalize_dropped_ids)}명")
    for pid in normalize_dropped_ids:
        print(f"❌ Dropped at normalization stage: {pid}")

    # 로그 기록
    with open(log_path, "a") as f:
        f.write(f"\n===== {split.upper()} =====\n")
        f.write(f"정규화까지 됐지만 슬라이싱 도중 제외된 환자 수: {len(dropped_ids)}명\n")
        for pid in dropped_ids:
            f.write(f"{pid}\n")
        f.write(f"정규화 도중 제외된 환자 수: {len(normalize_dropped_ids)}명\n")
        for pid in normalize_dropped_ids:
            f.write(f"{pid}\n")

print(f"\n전체 로그 저장 완료: {log_path}")