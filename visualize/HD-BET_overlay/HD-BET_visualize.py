

import os
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

# 경로 설정
original_path = "/Users/iujeong/0.local/original_data/all_t1c"
bet_path = "/Users/iujeong/0.local/1.bet_all/b_train"
save_path = "/Users/iujeong/0.local/visualize/HD-BET_overlay"

os.makedirs(save_path, exist_ok=True)

# 환자 리스트 불러오기
patient_ids = sorted(os.listdir(original_path))

def get_middle_slice(img_array):
    """Z 방향 중간 슬라이스 인덱스를 가져와서 axial 평면 반환"""
    z = img_array.shape[2] // 2
    return img_array[:, :, z]

for pid in patient_ids:
    if not pid.endswith(".nii.gz"):
        continue
    name = pid.replace(".nii.gz", "")
    orig_file = os.path.join(original_path, pid)
    bet_file = os.path.join(bet_path, name + "_bet.nii.gz")

    if not os.path.exists(bet_file):
        continue

    orig_nii = nib.load(orig_file)
    bet_nii = nib.load(bet_file)

    orig_img = orig_nii.get_fdata()
    bet_img = bet_nii.get_fdata()

    orig_slice = get_middle_slice(orig_img)
    bet_slice = get_middle_slice(bet_img)

    fig, axs = plt.subplots(1, 2, figsize=(8, 4))
    axs[0].imshow(orig_slice.T, cmap="gray", origin="lower")
    axs[0].set_title("Original")
    axs[1].imshow(bet_slice.T, cmap="gray", origin="lower")
    axs[1].set_title("HD-BET")
    for ax in axs:
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f"{name}_bet_vis.png"))
    plt.close()