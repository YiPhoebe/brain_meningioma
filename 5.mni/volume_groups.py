# scripts/volume_groups.py
import pandas as pd
IN = "/Users/iujeong/0.local/5.mni/location_by_patient.csv"
OUT = "/Users/iujeong/0.local/5.mni/volume_group_perf.csv"

df = pd.read_csv(IN)

# 사분위수 기반 구간 예시(팀 기준으로 바꿔도 됨: 예, S < 5cc, M 5~20cc, L >=20cc)
q1, q3 = df["vol_gt_cc"].quantile([0.33, 0.66])
def grp(v):
    if v < q1: return "Small"
    elif v < q3: return "Medium"
    else: return "Large"

df["vol_group"] = df["vol_gt_cc"].apply(grp)
out = df.groupby(["vol_group"])["dice"].agg(["count","mean","std"]).reset_index()
out.to_csv(OUT, index=False)
print("saved:", OUT)