# /Users/iujeong/0.local/scripts/locate_and_score.py
import os, json, glob, numpy as np, nibabel as nib, pandas as pd
from scipy.ndimage import center_of_mass

MNI_ATLAS  = "/Users/iujeong/0.local/mni/HarvardOxford-cort-maxprob-thr25-1mm.nii.gz"
LABELS_JSON= "/Users/iujeong/0.local/mni/HarvardOxford-labels.json"
LOBE_MAP_JSON = "/Users/iujeong/0.local/mni/LOBE_MAP.json"

IN_DIRS = [
  "/Users/iujeong/0.local/3.normalize/mni_mapped_train",
  "/Users/iujeong/0.local/3.normalize/mni_mapped_val",
  "/Users/iujeong/0.local/3.normalize/mni_mapped_test",
]
OUT_CSV = "/Users/iujeong/0.local/5.mni/location_by_patient.csv"

def load_maps():
    with open(LABELS_JSON,"r",encoding="utf-8") as f: labels = json.load(f)
    with open(LOBE_MAP_JSON,"r",encoding="utf-8") as f: lobe_map = json.load(f)
    idx2name = {i:n for i,n in enumerate(labels)}
    name2lobe = {}
    for L, names in lobe_map.items():
        for n in names:
            name2lobe[n.lower()] = L
    return idx2name, name2lobe

def dice(gt, pr):
    inter = np.logical_and(gt, pr).sum()
    s = gt.sum() + pr.sum()
    return 2*inter/s if s>0 else 0.0

def voxel_cc(img_nii):
    z = img_nii.header.get_zooms()[:3]
    return (z[0]*z[1]*z[2])/1000.0

atlas_nii = nib.load(MNI_ATLAS); atlas = atlas_nii.get_fdata().astype(np.int16)
idx2name, name2lobe = load_maps()

rows=[]
for IN in IN_DIRS:
    for p in glob.glob(os.path.join(IN, "*_gtv_mask_mni.nii.gz")):
        sid = os.path.basename(p).replace("_gtv_mask_mni.nii.gz","")
        img_p  = os.path.join(IN, f"{sid}_norm_mni.nii.gz")
        gt_p   = p
        pred_p = os.path.join(IN, f"{sid}_pred_mask_mni.nii.gz")
        if not (os.path.exists(img_p) and os.path.exists(pred_p)): 
            continue
        gt_nii = nib.load(gt_p); gt = gt_nii.get_fdata()>0
        pr_nii = nib.load(pred_p); pr = pr_nii.get_fdata()>0

        # Dice
        d = dice(gt, pr)

        # Volumes
        vcc = voxel_cc(gt_nii)
        vol_gt = gt.sum()*vcc
        vol_pr = pr.sum()*vcc
        vol_ratio = (vol_gt - vol_pr)/(vol_gt+1e-8)

        # Centroid→atlas→lobe
        if gt.sum()>0:
            cy, cx, cz = center_of_mass(gt)
            cy, cx, cz = map(lambda x:int(round(x)), (cy,cx,cz))
            lbl = atlas[cy, cx, cz] if 0<=cy<atlas.shape[0] and 0<=cx<atlas.shape[1] and 0<=cz<atlas.shape[2] else 0
        else:
            lbl = 0
        label_name = {0:"Background", **idx2name}.get(int(lbl), "Background")
        lobe = name2lobe.get(label_name.lower(), "Other")

        # Overlap main lobe(보조)
        overlap={}
        atlas_vals = atlas[np.where(gt)]
        u,c = np.unique(atlas_vals, return_counts=True)
        for ui,ci in zip(u,c):
            nm = idx2name.get(int(ui),"Background")
            lb = name2lobe.get(nm.lower(),"Other")
            overlap[lb]=overlap.get(lb,0)+int(ci)
        if overlap:
            main_lobe = max(overlap, key=overlap.get)
            main_frac = overlap[main_lobe]/(gt.sum()+1e-8)
        else:
            main_lobe, main_frac = "Other", 0.0

        rows.append(dict(
            subject_id=sid, split=os.path.basename(IN).replace("mni_mapped_",""),
            dice=d, vol_gt_cc=vol_gt, vol_pr_cc=vol_pr, vol_ratio=vol_ratio,
            centroid_label=label_name, centroid_lobe=lobe,
            overlap_main_lobe=main_lobe, overlap_main_frac=main_frac
        ))

df = pd.DataFrame(rows)
os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
df.to_csv(OUT_CSV, index=False)
print("saved:", OUT_CSV, "| n=", len(df))