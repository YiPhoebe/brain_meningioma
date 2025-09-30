# scripts/threshold_sweep.py
import os, glob, numpy as np, nibabel as nib, pandas as pd

IN_DIR = "data/mni_mapped"
THS = [0.3, 0.5, 0.8]  # 필요에 맞게
OUT = "data/mni_mapped/threshold_sweep.csv"

def dice(gt, pr):
    inter = np.logical_and(gt, pr).sum()
    s = gt.sum() + pr.sum()
    return 2*inter/s if s>0 else 0.0

rows=[]
for prob_p in sorted(glob.glob(os.path.join(IN_DIR,"*_pred_prob_mni.nii.gz"))):
    sid = os.path.basename(prob_p).replace("_pred_prob_mni.nii.gz","")
    gt_p = os.path.join(IN_DIR, f"{sid}_gt_mni.nii.gz")
    meta_p = os.path.join(IN_DIR, "location_by_patient.csv")  # 이전 단계 CSV
    if not os.path.exists(gt_p): continue

    prob = nib.load(prob_p).get_fdata().astype(np.float32)
    gt   = nib.load(gt_p).get_fdata()>0

    # centroid_lobe 정보 합치기
    # (성능만 뽑으려면 생략 가능)
    # 여기서는 파일 읽는 반복을 피하려고 아래에서 merge 해도 됨.

    for t in THS:
        pr = (prob >= t)
        rows.append({"subject_id": sid, "thr": t, "dice": dice(gt, pr)})

df = pd.DataFrame(rows)

# 위치 정보 merge
loc = pd.read_csv(os.path.join(IN_DIR, "location_by_patient.csv"))[["subject_id","centroid_lobe"]]
m = df.merge(loc, on="subject_id", how="left")
m.to_csv(OUT, index=False)

# 엽별/threshold별 평균
piv = m.groupby(["centroid_lobe","thr"])["dice"].agg(["count","mean","std"]).reset_index()
piv.to_csv(OUT.replace(".csv","_by_lobe.csv"), index=False)
print("saved threshold sweep.")