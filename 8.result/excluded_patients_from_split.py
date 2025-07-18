import pandas as pd

# 전체 500명 들어있는 통계 파일
stats_df = pd.read_csv("/Users/iujeong/0.local/8.result/csv/bet_bbox_stats.csv")

# split 환자 리스트
train = pd.read_csv("/Users/iujeong/0.local/8.result/csv/patients_train.csv", header=None, names=["PatientID"])
val = pd.read_csv("/Users/iujeong/0.local/8.result/csv/patients_val.csv", header=None, names=["PatientID"])
test = pd.read_csv("/Users/iujeong/0.local/8.result/csv/patients_test.csv", header=None, names=["PatientID"])

# 통합
split = pd.concat([train, val, test], ignore_index=True)

# 빠진 환자 찾기
all_patients = set(stats_df["patient_id"].unique())
split_patients = set(split["PatientID"].unique())

excluded = sorted(all_patients - split_patients)

# 저장
excluded_df = pd.DataFrame(excluded, columns=["ExcludedPatientID"])
excluded_df.to_csv("excluded_patients_from_split.csv", index=False)

print(f"제외된 환자 수: {len(excluded)}")
excluded_df.head()