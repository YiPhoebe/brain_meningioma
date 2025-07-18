import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob
from scipy.ndimage import center_of_mass
from collections import defaultdict

# Config
npy_dir = "/Users/iujeong/0.local/4.slice/s_test/npy"
save_dir = "/Users/iujeong/0.local/visualize/slice"
os.makedirs(save_dir, exist_ok=True)

mask_suffix = "_mask.npy"
region_map = {
    "Frontal": [],
    "Occipital": [],
    "Temporal": [],
    "Cerebellum": [],
    "No Tumor": [],
}
region_patient = defaultdict(set)
slice_index_distribution = []

# Process all mask files
for mask_path in sorted(glob(os.path.join(npy_dir, f"*{mask_suffix}"))):
    mask = np.load(mask_path)
    fname = os.path.basename(mask_path)
    patient_id = fname.split('_slice_')[0]
    slice_part = fname.split('_slice_')[-1]
    slice_idx_str = slice_part.replace("_mask.npy", "")
    slice_idx = int(slice_idx_str)

    if np.sum(mask) == 0:
        region_map["No Tumor"].append(fname)
        continue

    com = center_of_mass(mask)
    h, w = mask.shape
    if com[0] < h * 0.3:
        region = "Frontal"
    elif com[0] > h * 0.7:
        region = "Occipital"
    elif h * 0.4 < com[0] < h * 0.6:
        region = "Temporal"
    else:
        region = "Cerebellum"

    region_map[region].append(fname)
    region_patient[region].add(patient_id)
    slice_index_distribution.append(slice_idx)

    # Save example slice if first one for this region
    if len(region_map[region]) == 1:
        img_path = mask_path.replace("_mask.npy", ".npy")
        if os.path.exists(img_path):
            img = np.load(img_path)
            overlay = (img * 0.7 + mask * 0.3 * img.max()).astype(np.uint8)
            plt.imsave(os.path.join(save_dir, f"{region}_example.png"), overlay, cmap='gray')

# Count by region (patients)
region_patient_count = {k: len(v) for k, v in region_patient.items()}
df = pd.DataFrame(list(region_patient_count.items()), columns=["Region", "PatientCount"])

# Barplot
plt.figure(figsize=(6, 4))
sns.barplot(x="Region", y="PatientCount", data=df)
plt.title("Patient Count by Tumor Location")
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "/Users/iujeong/0.local/visualize/slice/tumor_location_count.png"))
plt.close()

# Heatmap of axial slice distribution
plt.figure(figsize=(5, 8))
axial_bins = np.bincount(slice_index_distribution)
sns.heatmap(axial_bins[:, np.newaxis], cmap="magma", cbar=True, yticklabels=10)
plt.ylabel("Axial Slice Index")
plt.xticks([])
plt.title("Tumor Slice Distribution (Axial)")
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "axial_distribution_heatmap.png"))
plt.close()