# scripts/aggregate_by_lobe.py
import pandas as pd

IN_CSV = "data/mni_mapped/location_by_patient.csv"
OUT_SUM = "data/mni_mapped/lobe_performance_summary.csv"

df = pd.read_csv(IN_CSV)

def summarize(by):
    g = df.groupby(by)["dice"]
    s = g.agg(["count","mean","std"]).reset_index()
    s = s.sort_values("mean", ascending=False)
    return s

sum_centroid = summarize("centroid_lobe")
sum_overlap  = summarize("overlap_main_lobe")

sum_centroid.to_csv(OUT_SUM.replace(".csv","_centroid.csv"), index=False)
sum_overlap.to_csv(OUT_SUM.replace(".csv","_overlap.csv"), index=False)
print("saved summaries.")