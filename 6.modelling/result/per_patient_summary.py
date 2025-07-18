import pandas as pd

slice_csv = "/Users/iujeong/0.local/4.slice/check_code/slice_count_diff.csv"
metric_csv = "/Users/iujeong/0.local/6.modelling/result/csv/per_patient_metrics.csv"
save_path = "/Users/iujeong/0.local/6.modelling/result/csv/per_patient_summary.csv"

# 슬라이스 수 로딩
df_slice = pd.read_csv(slice_csv)

# 성능 지표 로딩
df_metric = pd.read_csv(metric_csv)

# 병합
df_merge = pd.merge(df_slice, df_metric, on="patient_id", how="left")

# 저장
df_merge.to_csv(save_path, index=False)
print(f"✅ 저장 완료: {save_path}")