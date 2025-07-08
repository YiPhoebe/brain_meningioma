import os
import numpy as np
import nibabel as nib
import pandas as pd
from glob import glob
from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2

LOG_ROOT = "/Users/iujeong/0.local/8.result/log"

bbox_df = pd.read_csv("/Users/iujeong/0.local/8.result/csv/bbox_stats.csv")
crop_sizes = [(160, 192)]


def crop_and_resize_to_size(slice_, x_min, x_max, y_min, y_max, target_shape=(160, 192)):
    crop = slice_[x_min:x_max, y_min:y_max]
    th, tw = target_shape
    resized = cv2.resize(crop, (tw, th), interpolation=cv2.INTER_LINEAR)
    return resized

def save_slices(input_dir, output_dir, phase):
    nii_files = sorted(glob(os.path.join(input_dir, "*.nii.gz")))
    os.makedirs(output_dir, exist_ok=True)

    log_file = os.path.join(LOG_ROOT, f"filtered_{phase}.log")

    for path in tqdm(nii_files):
        pid = os.path.basename(path).split("_")[0]
        image = nib.load(path).get_fdata()
        gtv_path = path.replace("_t1c_bet.nii.gz", "_t1c_gtv_mask.nii.gz")
        gtv_mask = nib.load(gtv_path).get_fdata()

        bbox = bbox_df[(bbox_df["patient_id"] == pid) & (bbox_df["phase"] == phase)]
        if bbox.empty:
            with open(log_file, "a") as log:
                log.write(f"{pid} - bbox 정보 없음, 슬라이스 건너뜀\n")
            continue
        x_min, x_max = int(bbox["x_min"]), int(bbox["x_max"])
        y_min, y_max = int(bbox["y_min"]), int(bbox["y_max"])

        for crop_size in crop_sizes:
            h, w = crop_size
            npy_dir = os.path.join(output_dir, f"npy_{h}x{w}")
            png_dir = os.path.join(output_dir, f"png_{h}x{w}")
            os.makedirs(npy_dir, exist_ok=True)
            os.makedirs(png_dir, exist_ok=True)

        for i in range(image.shape[2]):
            try:
                slice_ = image[:, :, i]
                mask_slice = gtv_mask[:, :, i]
                for crop_size in crop_sizes:
                    cropped = crop_and_resize_to_size(slice_, x_min, x_max, y_min, y_max, target_shape=crop_size)
                    cropped_mask = crop_and_resize_to_size(mask_slice, x_min, x_max, y_min, y_max, target_shape=crop_size)
                    if np.sum(cropped_mask) < 10:
                        continue  # GTV가 너무 작으면 저장하지 않음
                    h, w = crop_size
                    npy_path = os.path.join(output_dir, f"npy_{h}x{w}", f"{pid}_{i:03d}.npy")
                    mask_path = os.path.join(output_dir, f"npy_{h}x{w}", f"{pid}_{i:03d}_mask.npy")
                    png_path = os.path.join(output_dir, f"png_{h}x{w}", f"{pid}_{i:03d}.png")
                    np.save(npy_path, cropped)
                    np.save(mask_path, cropped_mask)
                    plt.imsave(png_path, cropped, cmap='gray')

            except Exception as e:
                with open(os.path.join(LOG_ROOT, "failed_slices.log"), "a") as fail_log:
                    fail_log.write(f"{pid} - 슬라이스 {i}: 저장 실패 → {str(e)}\n")
                with open(log_file, "a") as log:
                    log.write(f"{pid} - 슬라이스 {i}: 예외 발생 → {str(e)}\n")

        with open(log_file, "a") as log:
            log.write(f"✅ {pid}: {image.shape[2]}개 슬라이스 저장 완료 (✅ bbox 중심 기준 crop)\n")

save_slices("/Users/iujeong/0.local/3.normalize/n_train/nii", "/Users/iujeong/0.local/4.slice/s_train", "train")
save_slices("/Users/iujeong/0.local/3.normalize/n_val/nii", "/Users/iujeong/0.local/4.slice/s_val", "val")
save_slices("/Users/iujeong/0.local/3.normalize/n_test/nii", "/Users/iujeong/0.local/4.slice/s_test", "test")