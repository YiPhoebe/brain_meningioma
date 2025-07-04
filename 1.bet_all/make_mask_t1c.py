#  t1c_bet.nii.gz 파일에 대해 binary 마스크(_bet_mask.nii.gz)를 만드는 코드
# 저장위치 :
# /Users/iujeong/0.local/1.bet_all/b_test
# /Users/iujeong/0.local/1.bet_all/b_train
# /Users/iujeong/0.local/1.bet_all/b_val

import nibabel as nib
import numpy as np
import os

for bet_dir in [
    "/Users/iujeong/0.local/1.bet_all/b_test",
    "/Users/iujeong/0.local/1.bet_all/b_train",
    "/Users/iujeong/0.local/1.bet_all/b_val",
]:
    for f in os.listdir(bet_dir):
        if f.endswith("_bet.nii.gz") and not f.startswith("."):
            filepath = os.path.join(bet_dir, f)
            try:
                img = nib.load(filepath)
            except Exception as e:
                print(f"Error loading {filepath}: {e}")
                continue

            out_path = filepath.replace("_bet.nii.gz", "_bet_mask.nii.gz")
            if os.path.exists(out_path):
                continue

            data = img.get_fdata()

            mask = (data > 0).astype(np.uint8)
            mask_img = nib.Nifti1Image(mask, img.affine)

            nib.save(mask_img, out_path)