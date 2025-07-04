import os
import numpy as np
import pandas as pd
import imageio
from scipy.stats import zscore

input_base = "/Users/iujeong/0.local/3.normalize"
out_base = "/Users/iujeong/0.local/4.slice"

def pad_slice(slice, x, y, pad_size=None):
    if pad_size is None:
        raise ValueError("pad_size must be specified")
    h, w = slice.shape
    half_pad = pad_size // 2

    x1 = int(x - half_pad)
    y1 = int(y - half_pad)
    x2 = x1 + pad_size
    y2 = y1 + pad_size

    pad_x1 = max(0, -x1)
    pad_y1 = max(0, -y1)
    pad_x2 = max(0, x2 - w)
    pad_y2 = max(0, y2 - h)

    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w, x2)
    y2 = min(h, y2)

    cropped = slice[y1:y2, x1:x2]
    padded = np.pad(cropped, ((pad_y1, pad_y2), (pad_x1, pad_x2)), mode='constant', constant_values=0)
    return padded

def save_filtered_slices(vol, mask, patient_id, group):
    img_dir = os.path.join(input_base, f"n_{group}/nii2")
    npy_out = os.path.join(out_base, f"s_{group}/npy2")
    png_out = os.path.join(out_base, f"s_{group}/png2")

    os.makedirs(npy_out, exist_ok=True)
    os.makedirs(png_out, exist_ok=True)

    global_mean = np.mean(vol)
    global_std = np.std(vol)

    keep_idx = filter_slices_by_mask_area(mask, patient_id=patient_id, group=group)

    bbox_csv="/Users/iujeong/0.local/8.result/csv/bbox_stats2.csv"
    bbox_df = pd.read_csv(bbox_csv)
    bbox = bbox_df[bbox_df["patient_id"] == patient_id]
    if bbox.empty:
        print(f"⚠️ {patient_id}: bbox stats missing")
        x_center = vol.shape[1] // 2
        y_center = vol.shape[0] // 2
        pad_size = 128
    else:
        x_min = int(bbox["x_min"].values[0])
        x_max = int(bbox["x_max"].values[0])
        y_min = int(bbox["y_min"].values[0])
        y_max = int(bbox["y_max"].values[0])
        x_center = (x_min + x_max) // 2
        y_center = (y_min + y_max) // 2
        width = int(x_max - x_min)
        height = int(y_max - y_min)
        pad_size = int(np.ceil(max(width, height) + 16))  # add margin

    out_npy_dir = os.path.join(npy_out, patient_id)
    out_png_dir = os.path.join(png_out, patient_id)
    os.makedirs(out_npy_dir, exist_ok=True)
    os.makedirs(out_png_dir, exist_ok=True)

    for i in keep_idx:
        vol_slice = vol[:, :, i]
        mask_slice = mask[:, :, i]

        vol_slice_padded = pad_slice(vol_slice, x_center, y_center, pad_size=pad_size)
        mask_slice_padded = pad_slice(mask_slice, x_center, y_center, pad_size=pad_size)

        np.save(os.path.join(out_npy_dir, f"{patient_id}_slice_{i:03d}_img.npy"), vol_slice_padded)
        np.save(os.path.join(out_npy_dir, f"{patient_id}_slice_{i:03d}_mask.npy"), mask_slice_padded)

        imageio.imwrite(os.path.join(out_png_dir, f"{patient_id}_slice_{i:03d}_img.png"), (vol_slice_padded * 255).astype(np.uint8))
        imageio.imwrite(os.path.join(out_png_dir, f"{patient_id}_slice_{i:03d}_mask.png"), (mask_slice_padded * 255).astype(np.uint8))

def filter_slices_by_mask_area(
    masks: np.ndarray,
    patient_id: str,
    group: str = None,
    bbox_csv="/Users/iujeong/0.local/8.result/csv/bbox_stats2.csv",
    clip_csv="/Users/iujeong/0.local/8.result/csv/gtv_clipping_stats2.csv",
    area_thresh: int = 10,
    z_thresh: float = 2.5,
):
    bbox_df = pd.read_csv(bbox_csv)
    clip_df = pd.read_csv(clip_csv)

    bbox = bbox_df[bbox_df["patient_id"] == patient_id]
    clip = clip_df[clip_df["patient_id"] == patient_id]
    if bbox.empty or clip.empty:
        print(f"⚠️ {patient_id}: bbox or clipping stats missing")
        if group is not None:
            log_file = os.path.join("/Users/iujeong/0.local/8.result/log", f"filtered_{group}2.log")
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            with open(log_file, "a") as f:
                f.write(f"{patient_id}: bbox or clipping stats missing\n")

    areas = []
    for i in range(masks.shape[2]):
        area = np.sum(masks[:, :, i] > 0)
        areas.append(area)

    areas = np.array(areas)
    keep_idx = np.where(areas > area_thresh)[0]

    if len(keep_idx) == 0:
        return []

    areas_kept = areas[keep_idx]
    areas_z = zscore(areas_kept)
    filtered_idx = keep_idx[np.abs(areas_z) <= z_thresh]

    return filtered_idx.tolist()


if __name__ == "__main__":
    import nibabel as nib
    import glob

    groups = ["train", "val", "test"]
    for group in groups:
        nii_dir = os.path.join(input_base, f"n_{group}/nii2")
        norm_files = glob.glob(os.path.join(nii_dir, "*_norm.nii.gz"))

        for norm_file in norm_files:
            basename = os.path.basename(norm_file)
            if not basename.endswith("_norm.nii.gz"):
                continue
            patient_id = basename.replace("_norm.nii.gz", "")
            resample_dir = os.path.join("/Users/iujeong/0.local/2.resample", f"r_{group}")
            mask_file = os.path.join(resample_dir, f"{patient_id}_t1c_gtv_mask.nii.gz")

            if not os.path.exists(mask_file):
                print(f"❌ 파일 없음: {mask_file}")
                continue

            vol = nib.load(norm_file).get_fdata()
            mask = nib.load(mask_file).get_fdata()
            save_filtered_slices(vol, mask, patient_id, group)
            print(f"✅ {patient_id} 저장 완료")
