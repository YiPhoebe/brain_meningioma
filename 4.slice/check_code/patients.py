
from glob import glob
import os
from collections import defaultdict

# 슬라이스 파일 경로
all_slices = sorted(glob("/Users/iujeong/0.local/4.slice/s_*/npy/*_img.npy"))

# 환자별로 카운트
patient_slice_count = defaultdict(int)
for path in all_slices:
    pid = os.path.basename(path).split("_slice_")[0]
    patient_slice_count[pid] += 1

# 총 슬라이스 수
total_slices = sum(patient_slice_count.values())
num_patients = len(patient_slice_count)
avg_slices = total_slices / num_patients

print(f"총 환자 수: {num_patients}")
print(f"총 슬라이스 수: {total_slices}")
print(f"1명당 평균 슬라이스 수: {avg_slices:.1f}")

# 요약 결과 저장
import pandas as pd

summary_df = pd.DataFrame({
    "총 환자 수": [num_patients],
    "총 슬라이스 수": [total_slices],
    "평균 슬라이스 수": [round(avg_slices, 1)]
})

save_path = "/Users/iujeong/0.local/4.slice/check_code/patients.csv"
summary_df.to_csv(save_path, index=False)
print(f"📁 요약 저장 완료: {save_path}")