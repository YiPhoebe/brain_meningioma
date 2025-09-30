from glob import glob
import os
import pandas as pd

# split 기준 환자 리스트 경로 정의
split_dir = "/Users/iujeong/0.local/split"


# ✅ 기존 CSV 파일에서 split 기준 expected 파일 생성
csv_based_ids = {
    "s_train": "/Users/iujeong/0.local/4.slice/check_code/s_train_patient_ids.csv",
    "s_val": "/Users/iujeong/0.local/4.slice/check_code/s_val_patient_ids.csv",
    "s_test": "/Users/iujeong/0.local/4.slice/check_code/s_test_patient_ids.csv",
}

# split 디렉토리 없으면 생성
os.makedirs(split_dir, exist_ok=True)

for split_name, csv_path in csv_based_ids.items():
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        expected_txt_path = os.path.join(split_dir, f"{split_name}_expected.txt")
        with open(expected_txt_path, "w") as f:
            for pid in sorted(df["patient_id"].tolist()):
                f.write(pid + "\n")
        print(f"✅ {split_name}_expected.txt 생성 완료 ({len(df)}명)")
    else:
        print(f"[경고] {csv_path} 파일 없음")

# 전체 환자 리스트 (원본 500명 기준)
total_patient_ids = set([
    f"BraTS-MEN-RT-{i:04d}-1" for i in range(500)
])

for phase in ["s_test", "s_train", "s_val"]:
    path = f"/Users/iujeong/0.local/4.slice/{phase}/npy"
    files = sorted(glob(os.path.join(path, "*_img.npy"), recursive=True))

    if not files:
        print(f"[WARN] {phase}에 해당하는 파일 없음! 경로 확인 필요: {path}")
    else:
        patient_ids = set(os.path.basename(f).split("_slice_")[0] for f in files)
        print(f"{phase}: {len(patient_ids)}명 | 총 슬라이스: {len(files)}")

        # 저장
        df = pd.DataFrame({"patient_id": sorted(patient_ids)})
        df.to_csv(f"{phase}_patient_ids.csv", index=False)

        # 누락 환자 체크
        removed = sorted(total_patient_ids - patient_ids)
        if removed:
            print(f"❌ {phase}에서 제외된 환자 수: {len(removed)}명")
            for r in removed:
                print(f"   - {r}")

# ✅ 모든 split 환자 ID 모아서 진짜 전처리 제외된 환자 찾기
all_present_ids = set()

for phase in ["s_test", "s_train", "s_val"]:
    path = f"/Users/iujeong/0.local/4.slice/{phase}/npy"
    files = sorted(glob(os.path.join(path, "*_img.npy"), recursive=True))
    patient_ids = set(os.path.basename(f).split("_slice_")[0] for f in files)
    all_present_ids |= patient_ids  # 합집합으로 누적

excluded_ids = sorted(total_patient_ids - all_present_ids)
print(f"\n✅ 전처리에서 실제로 제외된 환자 수: {len(excluded_ids)}명")
for pid in excluded_ids:
    print(f"   - {pid}")

# CSV로 저장
excluded_df = pd.DataFrame({"excluded_patient_id": excluded_ids})
excluded_df.to_csv("/Users/iujeong/0.local/4.slice/check_code/excluded_patients.csv", index=False)


# ✅ split 기준과 실제 전처리된 환자 비교
for split_name in ["s_train", "s_val", "s_test"]:
    expected_path = os.path.join(split_dir, f"{split_name}_expected.txt")
    actual_path = f"{split_name}_patient_ids.csv"

    if not os.path.exists(expected_path) or not os.path.exists(actual_path):
        print(f"[SKIP] {split_name}: 비교할 파일 없음")
        continue

    expected_ids = set(line.strip() for line in open(expected_path))
    actual_ids = set(pd.read_csv(actual_path)["patient_id"].tolist())

    missing = sorted(expected_ids - actual_ids)
    extra = sorted(actual_ids - expected_ids)

    print(f"\n🔎 [{split_name}] 비교 결과")
    print(f" - 누락된 환자 수: {len(missing)}")
    if missing:
        for pid in missing:
            print(f"   ❌ 누락: {pid}")

    print(f" - 잘못 포함된 환자 수: {len(extra)}")
    if extra:
        for pid in extra:
            print(f"   ⚠️ 오버: {pid}")