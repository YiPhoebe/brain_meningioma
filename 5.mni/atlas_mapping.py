# scripts/atlas_mapping.py
import os, json, numpy as np, nibabel as nib
from nilearn import datasets

MNI_TEMPLATE = "data/mni/MNI152_T1_1mm.nii.gz"
OUT_ATLAS    = "data/mni/harvard_oxford_atlas_1mm.nii.gz"
OUT_LABELS   = "data/mni/harvard_oxford_labels.json"

# 2-1) 아틀라스 다운로드 (probabilistic max-prob으로 쓰기)
ho = datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr25-1mm')  # 1mm, max-prob, thr=25%
# ho.maps: Nifti1Image, ho.labels: list[str]
nib.save(ho.maps, OUT_ATLAS)
with open(OUT_LABELS, "w", encoding="utf-8") as f:
    json.dump(ho.labels, f, ensure_ascii=False, indent=2)

# 2-2) 엽 단위 매핑 사전(예시) — 필요시 세분화/수정
# Atlas별 label 이름 다름. ho.labels 예시 기준으로 매핑.
# *중요*: labels[0]은 'Background'인 경우가 많음.
LOBE_MAP = {
    "Frontal Pole": "Frontal", "Superior Frontal Gyrus": "Frontal",
    "Middle Frontal Gyrus": "Frontal", "Inferior Frontal Gyrus, pars triangularis": "Frontal",
    "Inferior Frontal Gyrus, pars opercularis": "Frontal", "Precentral Gyrus": "Frontal",
    "Frontal Medial Cortex": "Frontal", "Frontal Orbital Cortex": "Frontal",

    "Postcentral Gyrus": "Parietal", "Superior Parietal Lobule": "Parietal",
    "Supramarginal Gyrus, anterior division": "Parietal",
    "Supramarginal Gyrus, posterior division": "Parietal",
    "Angular Gyrus": "Parietal", "Precuneus Cortex": "Parietal",

    "Temporal Pole": "Temporal", "Superior Temporal Gyrus, anterior division": "Temporal",
    "Superior Temporal Gyrus, posterior division": "Temporal",
    "Middle Temporal Gyrus, anterior division": "Temporal",
    "Middle Temporal Gyrus, posterior division": "Temporal",
    "Inferior Temporal Gyrus, anterior division": "Temporal",
    "Inferior Temporal Gyrus, posterior division": "Temporal",
    "Heschl's Gyrus (includes H1 and H2)": "Temporal",
    "Planum Temporale": "Temporal", "Temporal Occipital Fusiform Cortex": "Temporal",
    "Temporal Fusiform Cortex, anterior division": "Temporal",
    "Temporal Fusiform Cortex, posterior division": "Temporal",

    "Lateral Occipital Cortex, superior division": "Occipital",
    "Lateral Occipital Cortex, inferior division": "Occipital",
    "Occipital Pole": "Occipital",

    "Cingulate Gyrus, anterior division": "Cingulate",
    "Cingulate Gyrus, posterior division": "Cingulate",
    "Paracingulate Gyrus": "Cingulate",

    "Insular Cortex": "Insula",
}
# Cerebellum/Deep nuclei는 이 atlas에 제한적. 필요하면 SUIT/AAL 등 추가로 합쳐서 사용.
with open("data/mni/LOBE_MAP.json","w",encoding="utf-8") as f:
    json.dump(LOBE_MAP, f, ensure_ascii=False, indent=2)