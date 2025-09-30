# /Users/iujeong/0.local/scripts/aggregate_reports.py
import os, pandas as pd, numpy as np, nibabel as nib, glob

IN_CSV = "/Users/iujeong/0.local/5.mni/location_by_patient.csv"
OUT_DIR= "/Users/iujeong/0.local/5.mni/reports"
os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_csv(IN_CSV)

# 3-1) 엽별 성능 (centroid 기준)
centroid = df.groupby("centroid_lobe")["dice"].agg(["count","mean","std"]).reset_index().sort_values("mean", ascending=False)
centroid.to_csv(os.path.join(OUT_DIR,"lobe_perf_centroid.csv"), index=False)

# 3-2) 엽별 성능 (overlap 기준)
overlap = df.groupby("overlap_main_lobe")["dice"].agg(["count","mean","std"]).reset_index().sort_values("mean", ascending=False)
overlap.to_csv(os.path.join(OUT_DIR,"lobe_perf_overlap.csv"), index=False)

# 3-3) 부피 그룹별 성능
q1, q3 = df["vol_gt_cc"].quantile([0.33, 0.66])
def grp(v): 
    return "Small" if v<q1 else ("Medium" if v<q3 else "Large")
df["vol_group"] = df["vol_gt_cc"].apply(grp)
vg = df.groupby("vol_group")["dice"].agg(["count","mean","std"]).reset_index()
vg.to_csv(os.path.join(OUT_DIR,"volume_group_perf.csv"), index=False)

print("saved reports in", OUT_DIR)

# 3-4) (옵션) threshold 스윕: pred_prob가 있을 때
# 각 split의 MNI 폴더들
MNI_DIRS = [
  "/Users/iujeong/0.local/3.normalize/mni_mapped_train",
  "/Users/iujeong/0.local/3.normalize/mni_mapped_val",
  "/Users/iujeong/0.local/3.normalize/mni_mapped_test",
]
THS = [0.3, 0.5, 0.8]

rows=[]
for d in MNI_DIRS:
    for prob in glob.glob(os.path.join(d,"*_pred_prob_mni.nii.gz")):
        sid = os.path.basename(prob).replace("_pred_prob_mni.nii.gz","")
        gt_p = os.path.join(d, f"{sid}_gtv_mask_mni.nii.gz")
        if not os.path.exists(gt_p): 
            continue
        pr = nib.load(prob).get_fdata().astype("float32")
        gt = nib.load(gt_p).get_fdata()>0
        for t in THS:
            pm = pr>=t
            inter = (pm & gt).sum()
            s = pm.sum()+gt.sum()
            dice = (2*inter/s) if s>0 else 0.0
            rows.append({"subject_id":sid,"thr":t,"dice":dice})

if rows:
    import pandas as pd
    thr = pd.DataFrame(rows)
    thr = thr.merge(df[["subject_id","centroid_lobe"]], on="subject_id", how="left")
    out = thr.groupby(["centroid_lobe","thr"])["dice"].agg(["count","mean","std"]).reset_index()
    out.to_csv(os.path.join(OUT_DIR,"threshold_sweep_by_lobe.csv"), index=False)
    print("saved threshold sweep")
