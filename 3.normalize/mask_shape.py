import os
import numpy as np
import nibabel as nib
from glob import glob
from tqdm import tqdm
import pandas as pd
from scipy.ndimage import zoom

def resize_mask_to_match(target_shape, mask, order=0):
    scale_factors = np.array(target_shape) / np.array(mask.shape)
    return zoom(mask, scale_factors, order=order)  # nearest for mask

def process_masks(norm_img_dir, bet_mask_dir, gtv_mask_dir, save_dir, csv_path, split='train'):
    os.makedirs(save_dir, exist_ok=True)
    records = []

    norm_paths = sorted(glob(os.path.join(norm_img_dir, "*_norm.nii.gz")))  # ✅ 이걸로 고쳐

    for path in tqdm(norm_paths):
        name = os.path.basename(path).replace("_norm.nii.gz", "")
        print("🔍 Processing:", name)
        bet_path = os.path.join(bet_mask_dir, f"{name}_t1c_bet_mask.nii.gz")
        gtv_path = os.path.join(gtv_mask_dir, f"{name}_t1c_gtv_mask.nii.gz")
        print("  BET path exists:", os.path.exists(bet_path), "->", bet_path)
        print("  GTV path exists:", os.path.exists(gtv_path), "->", gtv_path)

        img = nib.load(path).get_fdata()
        shape = img.shape  # normalized image shape

        # Load corresponding masks
        if not os.path.exists(bet_path) or not os.path.exists(gtv_path):
            print(f"Missing mask for {name}, skipping")
            continue

        bet = nib.load(bet_path).get_fdata()
        gtv = nib.load(gtv_path).get_fdata()

        bet_resized = resize_mask_to_match(shape, bet)
        gtv_resized = resize_mask_to_match(shape, gtv)

        # 저장
        nib.save(nib.Nifti1Image(bet_resized, affine=np.eye(4)), os.path.join(save_dir, f"{name}_bet_mask.nii.gz"))
        nib.save(nib.Nifti1Image(gtv_resized, affine=np.eye(4)), os.path.join(save_dir, f"{name}_gtv_mask.nii.gz"))

        # 슬라이스 단위 위치 기록
        for z in range(shape[2]):
            gtv_slice = gtv_resized[:, :, z]
            if np.sum(gtv_slice) == 0:
                continue  # 병변 없는 슬라이스는 스킵

            ys, xs = np.where(gtv_slice > 0)
            x_min, x_max = xs.min(), xs.max()
            y_min, y_max = ys.min(), ys.max()

            records.append([name, split, z, x_min, x_max, y_min, y_max])

    # CSV 저장
    df = pd.DataFrame(records, columns=["case", "split", "z", "x_min", "x_max", "y_min", "y_max"])
    df.to_csv(csv_path, index=False)

    
def process_all_sets():
    base_bet = "/Users/iujeong/0.local/2.resample"
    base_gtv = "/Users/iujeong/0.local/2.resample"
    base_norm = "/Users/iujeong/0.local/3.normalize"

    for split in ["train", "val", "test"]:
        norm_img_dir = os.path.join(base_norm, f"n_{split}")
        bet_mask_dir = os.path.join(base_bet, f"r_{split}")
        gtv_mask_dir = os.path.join(base_gtv, f"r_{split}")
        save_dir = os.path.join(base_norm, f"n_{split}")
        csv_path = os.path.join(base_norm, f"mask_slices.csv")

        process_masks(norm_img_dir, bet_mask_dir, gtv_mask_dir, save_dir, csv_path, split)



if __name__ == "__main__":
    process_all_sets()