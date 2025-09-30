
region_colors = {
    "Frontal": "blue",
    "Parietal": "green",
    "Temporal_L": "orange",
    "Temporal_R": "red",
    "Occipital": "purple",
    "Cerebellum_L": "cyan",
    "Cerebellum_R": "magenta"
}

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


csv_path = "/Users/iujeong/0.local/4.slice/tumor_centers.csv"

img_path = "/Users/iujeong/0.local/4.slice/s_test/npy/BraTS-MEN-RT-0145-1_slice_108_img.npy"
save_root = "/Users/iujeong/0.local/4.slice/tumor_centers_img"
os.makedirs(save_root, exist_ok=True)

df = pd.read_csv(csv_path)
df_slice = df[df["MaskPath"].str.contains("BraTS-MEN-RT-0145-1_slice_108")]

img = np.load(img_path)
fig, ax = plt.subplots(figsize=(4, 4))
ax.imshow(img, cmap="gray")

# Draw rectangles for each region in df_slice
for _, row in df_slice.iterrows():
    region = row["Region"]
    color = region_colors.get(region, "gray")

    if region == "Frontal":
        rect = plt.Rectangle((0, 0), 128, 50, linewidth=1.5, edgecolor=color, facecolor=color, alpha=0.2)
    elif region == "Parietal":
        rect = plt.Rectangle((40, 50), 48, 80, linewidth=1.5, edgecolor=color, facecolor=color, alpha=0.2)
    elif region == "Temporal_L":
        rect = plt.Rectangle((0, 50), 40, 80, linewidth=1.5, edgecolor=color, facecolor=color, alpha=0.2)
    elif region == "Temporal_R":
        rect = plt.Rectangle((88, 50), 40, 80, linewidth=1.5, edgecolor=color, facecolor=color, alpha=0.2)
    elif region == "Occipital":
        rect = plt.Rectangle((40, 130), 48, 98, linewidth=1.5, edgecolor=color, facecolor=color, alpha=0.2)
    elif region == "Cerebellum_L":
        rect = plt.Rectangle((0, 130), 40, 98, linewidth=1.5, edgecolor=color, facecolor=color, alpha=0.2)
    elif region == "Cerebellum_R":
        rect = plt.Rectangle((88, 130), 40, 98, linewidth=1.5, edgecolor=color, facecolor=color, alpha=0.2)
    else:
        rect = plt.Rectangle((0, 0), 128, 228, linewidth=1.5, edgecolor="gray", facecolor="gray", alpha=0.1)

    ax.add_patch(rect)

ax.axis("off")

save_path = os.path.join(save_root, "BraTS-MEN-RT-0145-1_slice_108.png")
plt.savefig(save_path, bbox_inches="tight", pad_inches=0)
plt.close()
