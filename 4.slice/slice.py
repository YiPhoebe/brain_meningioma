def crop_to_brain_region(img, mask):
    coords = np.argwhere(mask > 0)
    if coords.size == 0:
        return None  # or raise error?
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0) + 1
    return img[y_min:y_max, x_min:x_max]

# Crop a 3D volume to the bounding box of the brain region in bet_mask
def crop_volume_to_brain(volume, bet_mask):
    coords = np.argwhere(bet_mask > 0)
    if coords.size == 0:
        return None
    zmin, ymin, xmin = coords.min(axis=0)
    zmax, ymax, xmax = coords.max(axis=0) + 1
    return volume[zmin:zmax, ymin:ymax, xmin:xmax]
import os
import numpy as np
import nibabel as nib
import csv
from glob import glob
from scipy.ndimage import center_of_mass
from scipy.stats import zscore
from tqdm import tqdm
import imageio
import matplotlib.pyplot as plt


def extract_patient_id(filename):
    return filename.split('/')[-1].split('_')[0]


def filter_patient_by_csv(csv_path, threshold=0.4):
    exclude = []
    if not os.path.exists(csv_path):
        print(f"[WARNING] CSV not found: {csv_path}. Skipping exclusion.")
        return exclude
    with open(csv_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if float(row['removed_ratio']) > threshold:
                exclude.append(row['patient_id'])
    return exclude


def load_image_volume(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image not found: {path}")
    img = nib.load(path)
    img = nib.as_closest_canonical(img)
    return img.get_fdata(), img.affine


def load_gtv_mask(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"GTV mask not found: {path}")
    img = nib.load(path)
    img = nib.as_closest_canonical(img)
    return img.get_fdata().astype(np.uint8), img.affine


def load_bet_mask(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"BET mask not found: {path}")
    img = nib.load(path)
    img = nib.as_closest_canonical(img)
    return img.get_fdata().astype(np.uint8), img.affine


def filter_slices_by_mask_area(gtv_mask, area_thresh=10, z_thresh=2.0):
    areas = np.sum(gtv_mask, axis=(1, 2))
    z_scores = zscore(areas)

    keep_slices = []
    for z, area, z_val in zip(range(len(areas)), areas, z_scores):
        if area >= area_thresh and abs(z_val) < z_thresh:
            keep_slices.append(z)
    return keep_slices

def save_filtered_slices(output_root, patient_id, image, gtv_mask, bet_mask, keep_slices, norm_path):
    npy_dir = os.path.join(output_root, patient_id)
    os.makedirs(npy_dir, exist_ok=True)

    for z in keep_slices:
        # Skip if z is out of bounds for any of the volumes
        if z >= image.shape[2] or z >= gtv_mask.shape[2] or z >= bet_mask.shape[2]:
            import logging
            logger = logging.getLogger("slice")
            logger.warning(f"[SKIP] {patient_id} - z={z} out of bounds (img:{image.shape[2]}, gtv:{gtv_mask.shape[2]}, bet:{bet_mask.shape[2]})")
            continue
        img_slice = image[:, :, z]
        mask_slice = gtv_mask[:, :, z]
        bet_slice = bet_mask[:, :, z]  # BET는 여전히 불러오지만 이미지에는 적용 안 함

        img_slice[bet_slice == 0] = 0  # Force background to zero before crop

        # Crop image and mask to brain region using BET mask
        img_slice = crop_to_brain_region(img_slice, bet_slice)
        mask_slice = crop_to_brain_region(mask_slice, bet_slice)

        if img_slice is None or mask_slice is None:
            with open("/Users/iujeong/0.local/8.result/log/skipped_empty_slices.log", "a") as f:
                f.write(f"{patient_id} slice {z:03d} skipped: empty BET region\n")
            continue

        desired_shape = (160, 192)
        h, w = img_slice.shape

        # Pad or crop image slice as needed
        if h > desired_shape[0]:
            img_slice = img_slice[:desired_shape[0], :]
            h = desired_shape[0]
        if w > desired_shape[1]:
            img_slice = img_slice[:, :desired_shape[1]]
            w = desired_shape[1]

        pad_h = (desired_shape[0] - h) // 2
        pad_w = (desired_shape[1] - w) // 2
        img_slice = np.pad(img_slice,
                           ((pad_h, desired_shape[0] - h - pad_h),
                            (pad_w, desired_shape[1] - w - pad_w)),
                           mode='constant', constant_values=0)

        h, w = mask_slice.shape

        # Pad or crop mask slice as needed
        if h > desired_shape[0]:
            mask_slice = mask_slice[:desired_shape[0], :]
            h = desired_shape[0]
        if w > desired_shape[1]:
            mask_slice = mask_slice[:, :desired_shape[1]]
            w = desired_shape[1]

        pad_h = (desired_shape[0] - h) // 2
        pad_w = (desired_shape[1] - w) // 2
        mask_slice = np.pad(mask_slice,
                            ((pad_h, desired_shape[0] - h - pad_h),
                             (pad_w, desired_shape[1] - w - pad_w)),
                            mode='constant', constant_values=0)

        # NOTE: 3D 볼륨은 이미 정규화되었으므로, 슬라이스 후에는 BET 적용하지 않고 저장함
        norm_slice = img_slice.astype(np.float32)
        norm_slice[np.isclose(norm_slice, 0.0, atol=1e-4)] = 0.0
        slice_id = f"{os.path.basename(norm_path).replace('_norm.nii.gz','')}_z{z:03}"
        np.save(os.path.join(npy_dir, f"{slice_id}_img.npy"), norm_slice)
        np.save(os.path.join(npy_dir, f"{slice_id}_mask.npy"), mask_slice)


if __name__ == "__main__":
    groups = ["train", "val", "test"]
    root_mask = "/Users/iujeong/0.local/2.resample"
    root_img = "/Users/iujeong/0.local/3.normalize"
    csv_dir = "/Users/iujeong/0.local/8.result/csv"
    exclude_csv = "/Users/iujeong/0.local/8.result/csv/gtv_clipping_stats.csv"

    for group in groups:
        output_root = f"/Users/iujeong/0.local/4.slice/s_{group}"
        print(f"\nProcessing group: {group}")
        norm_dir = os.path.join(root_img, f"n_{group}", "nii")
        mask_dir = os.path.join(root_mask, f"r_{group}")
        exclude = filter_patient_by_csv(exclude_csv)
        log_path = f"/Users/iujeong/0.local/8.result/filtered_{group}.log"
        os.makedirs(os.path.dirname(log_path), exist_ok=True)

        patient_paths = sorted(glob(os.path.join(norm_dir, "*_norm.nii.gz")))
        train_ids = [extract_patient_id(p) for p in patient_paths]

        location_records = []

        with open(log_path, "w") as logfile:
            for img_path in tqdm(patient_paths):
                patient_id = extract_patient_id(img_path)
                if patient_id in exclude:
                    logfile.write(f"[EXCLUDED] {patient_id} - high removed_ratio\n")
                    continue

                gtv_path = os.path.join(mask_dir, f"{patient_id}_t1c_gtv_mask.nii.gz")
                bet_path = os.path.join(mask_dir, f"{patient_id}_t1c_bet_mask.nii.gz")

                try:
                    img, _ = load_image_volume(img_path)
                    gtv, _ = load_gtv_mask(gtv_path)
                    bet, _ = load_bet_mask(bet_path)
                    if img.shape != gtv.shape or img.shape != bet.shape:
                        logfile.write(f"[MISMATCH] {patient_id} - shape mismatch: img{img.shape}, gtv{gtv.shape}, bet{bet.shape}\n")
                        continue
                except FileNotFoundError as e:
                    logfile.write(f"[MISSING] {patient_id} - {e}\n")
                    continue

                # Crop all 3D volumes to brain region using BET mask
                img = crop_volume_to_brain(img, bet)
                gtv = crop_volume_to_brain(gtv, bet)
                bet = crop_volume_to_brain(bet, bet)

                if img is None or gtv is None or bet is None:
                    logfile.write(f"[EMPTY] {patient_id} - empty BET region after cropping\n")
                    continue


                img[bet == 0] = 0
                img[np.isclose(img, 0.0, atol=1e-4)] = 0.0

                keep_slices = filter_slices_by_mask_area(gtv)

                if len(keep_slices) == 0:
                    logfile.write(f"[EMPTY] {patient_id} - no valid slices\n")
                    continue

                discontinuity = np.any(np.diff(keep_slices) > 1)
                if discontinuity:
                    logfile.write(f"[WARNING] {patient_id} - non-contiguous slices: {keep_slices}\n")

                save_filtered_slices(output_root, patient_id, img, gtv, bet, keep_slices, img_path)

                com = center_of_mass(gtv)
                location_records.append({"patient_id": patient_id, "z": int(com[2]), "y": int(com[1]), "x": int(com[0])})

        os.makedirs(csv_dir, exist_ok=True)
        csv_path = os.path.join(csv_dir, f"gtv_location_stats_{group}.csv")
        with open(csv_path, "w", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=["patient_id", "z", "y", "x"])
            writer.writeheader()
            writer.writerows(location_records)

    # 중복 검사
    if len(groups) >= 2:
        from itertools import combinations
        group_ids = {}
        for group in groups:
            norm_dir = os.path.join(root_img, f"n_{group}", "nii")
            paths = sorted(glob(os.path.join(norm_dir, "*_norm.nii.gz")))
            group_ids[group] = set(extract_patient_id(p) for p in paths)

        for a, b in combinations(groups, 2):
            dupes = group_ids[a] & group_ids[b]
            if dupes:
                print(f"[DUPLICATE WARNING] {a} ∩ {b} = {dupes}")