import os

# 경로 설정
normalized_dir = "/Users/iujeong/0.local/3.normalize/n_train"
sliced_dir = "/Users/iujeong/0.local/4.slice/s_train/npy"

# 환자 ID 추출 함수
def get_patient_ids_from_dir(directory, delimiter="_"):
    filenames = os.listdir(directory)
    patient_ids = set(f.split(delimiter)[0] for f in filenames if f.endswith(".nii.gz") or f.endswith(".npy"))
    return patient_ids

# 각 디렉토리에서 환자 ID 리스트 추출
normalized_ids = get_patient_ids_from_dir(normalized_dir)
sliced_ids = get_patient_ids_from_dir(sliced_dir)

# 슬라이싱에서 누락된 환자 찾기
dropped_ids = sorted(normalized_ids - sliced_ids)

# 결과 출력
print(f"정규화까지 됐지만 슬라이싱 도중 제외된 환자 수: {len(dropped_ids)}명")
for pid in dropped_ids:
    print(f"❌ Dropped at slicing stage: {pid}")

# 로그 저장
log_path = "/Users/iujeong/0.local/4.slice/check_code/slicing_dropped_cases.log"
with open(log_path, "w") as f:
    f.write(f"정규화까지 됐지만 슬라이싱 도중 제외된 환자 수: {len(dropped_ids)}명\n")
    for pid in dropped_ids:
        f.write(f"{pid}\n")
print(f"\n로그 저장 완료: {log_path}")