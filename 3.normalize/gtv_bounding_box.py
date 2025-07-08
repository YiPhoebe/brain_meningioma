
import csv



import numpy as np
import nibabel as nib
import os

gtv_dir = [
    "/Users/iujeong/0.local/3.normalize/n_test/nii",
    "/Users/iujeong/0.local/3.normalize/n_train/nii",
    "/Users/iujeong/0.local/3.normalize/n_val/nii"
]


def get_gtv_bounding_box(gtv_mask, margin=5):
    """Calculate bounding box around non-zero regions in the GTV mask."""
    if np.sum(gtv_mask) == 0:
        return None  # No tumor

    z_any, y_any, x_any = np.any(gtv_mask, axis=(1, 2)), np.any(gtv_mask, axis=(0, 2)), np.any(gtv_mask, axis=(0, 1))
    z_min, z_max = np.where(z_any)[0][[0, -1]]
    y_min, y_max = np.where(y_any)[0][[0, -1]]
    x_min, x_max = np.where(x_any)[0][[0, -1]]

    z_min = max(0, z_min - margin)
    z_max = min(gtv_mask.shape[0], z_max + margin)
    y_min = max(0, y_min - margin)
    y_max = min(gtv_mask.shape[1], y_max + margin)
    x_min = max(0, x_min - margin)
    x_max = min(gtv_mask.shape[2], x_max + margin)

    return (z_min, z_max, y_min, y_max, x_min, x_max)

from glob import glob

if __name__ == "__main__":
    results = []
    for dir_path in gtv_dir:
        nii_paths = glob(os.path.join(dir_path, "*_gtv_mask.nii.gz"))
        for path in sorted(nii_paths):
            gtv_nii = nib.load(path)
            gtv_mask = gtv_nii.get_fdata() > 0  # Binary mask

            bbox = get_gtv_bounding_box(gtv_mask)
            filename = os.path.basename(path)
            if bbox:
                results.append((filename, *bbox))
            else:
                results.append((filename, "No tumor", "", "", "", "", ""))

    os.makedirs("/Users/iujeong/0.local/8.result/csv", exist_ok=True)
    csv_path = "/Users/iujeong/0.local/8.result/csv/gtv_bounding_boxes.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "z_min", "z_max", "y_min", "y_max", "x_min", "x_max"])
        writer.writerows(results)

    print(f"Saved bounding boxes to {csv_path}")