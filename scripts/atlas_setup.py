# /Users/iujeong/0.local/scripts/atlas_setup.py
import os, json, nibabel as nib
from nilearn import datasets

OUT_DIR = "/Users/iujeong/0.local/mni"
os.makedirs(OUT_DIR, exist_ok=True)

# Harvard-Oxford cortex (max-prob, thr 25%, 1mm)
ho = datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr25-1mm')
nib.save(ho.maps, os.path.join(OUT_DIR, "HarvardOxford-cort-maxprob-thr25-1mm.nii.gz"))
labels = list(ho.labels)
with open(os.path.join(OUT_DIR, "HarvardOxford-labels.json"), "w", encoding="utf-8") as f:
    json.dump(labels, f, ensure_ascii=False, indent=2)

# 간단 엽 매핑(필요시 수정해서 다시 저장)
LOBE_MAP = {
    "Frontal": ["Frontal Pole","Superior Frontal Gyrus","Middle Frontal Gyrus","Inferior Frontal Gyrus, pars triangularis","Inferior Frontal Gyrus, pars opercularis","Precentral Gyrus","Frontal Medial Cortex","Frontal Orbital Cortex"],
    "Parietal": ["Postcentral Gyrus","Superior Parietal Lobule","Supramarginal Gyrus, anterior division","Supramarginal Gyrus, posterior division","Angular Gyrus","Precuneus Cortex"],
    "Temporal": ["Temporal Pole","Superior Temporal Gyrus, anterior division","Superior Temporal Gyrus, posterior division","Middle Temporal Gyrus, anterior division","Middle Temporal Gyrus, posterior division","Inferior Temporal Gyrus, anterior division","Inferior Temporal Gyrus, posterior division","Heschl's Gyrus (includes H1 and H2)","Planum Temporale","Temporal Occipital Fusiform Cortex","Temporal Fusiform Cortex, anterior division","Temporal Fusiform Cortex, posterior division"],
    "Occipital": ["Lateral Occipital Cortex, superior division","Lateral Occipital Cortex, inferior division","Occipital Pole"],
    "Cingulate": ["Cingulate Gyrus, anterior division","Cingulate Gyrus, posterior division","Paracingulate Gyrus"],
    "Insula": ["Insular Cortex"]
}
with open(os.path.join(OUT_DIR, "LOBE_MAP.json"), "w", encoding="utf-8") as f:
    json.dump(LOBE_MAP, f, ensure_ascii=False, indent=2)

print("saved atlas & maps in", OUT_DIR)
