import nibabel as nib
import numpy as np
import pandas as pd
import os
from glob import glob

# 루트 디렉토리 지정
nii_root = "/Users/iujeong/0.local/3.normalize/"
nan_locations = []

# 검사할 suffix 리스트
suffixes = ["_bet_mask.nii.gz", "_gtv_mask.nii.gz", "_norm.nii.gz"]

# n_train/nii, n_val/nii, n_test/nii 모두 검색
search_dirs = ["n_train/nii", "n_val/nii", "n_test/nii"]
for sub_dir in search_dirs:
    full_dir = os.path.join(nii_root, sub_dir)
    # 하위 폴더까지 모두 탐색
    nii_files = glob(os.path.join(full_dir, "**", "*.nii.gz"), recursive=True)
    for nii_file in nii_files:
        if not any(nii_file.endswith(sfx) for sfx in suffixes):
            continue
        try:
            volume = nib.load(nii_file).get_fdata()
            for z in range(volume.shape[2]):
                slice_2d = volume[:, :, z]
                if np.isnan(slice_2d).any():
                    yx_coords = np.argwhere(np.isnan(slice_2d))
                    for y, x in yx_coords:
                        nan_locations.append({
                            "file": os.path.basename(nii_file),
                            "z": z,
                            "y": y,
                            "x": x
                        })
        except Exception as e:
            print(f"❌ 파일 로딩 실패: {nii_file} → {e}")

df = pd.DataFrame(nan_locations)
df.to_csv("nan_coords.csv", index=False)
print("✅ NaN 좌표 저장 완료 → nan_coords.csv")