import os
import nibabel as nib
import subprocess
import glob
import pandas as pd
from sklearn.model_selection import train_test_split

import torch
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"📟 Using device: {device}")

# 경로 설정
input_dir = "/Users/iujeong/0.local/original_data/all_t1c"
gtv_dir = "/Users/iujeong/0.local/original_data/all_gtv"
log_path = os.path.join("/Users/iujeong/0.local/1.bet_all", "shape_log.txt")

# 모든 파일에서 base name 추출
all_files = sorted(glob.glob(os.path.join(input_dir, "*.nii.gz")))
all_ids = [os.path.basename(f).replace(".nii.gz", "") for f in all_files]

df_split = pd.read_csv("/Users/iujeong/0.local/2.resample/patient_ids_by_split.csv")
split_dict = dict(zip(df_split["patient_id"], df_split["split"]))  # 'r_train' 등

with open(log_path, "w") as log_f:
    for nii_path in all_files:
        filename = os.path.basename(nii_path).replace(".nii.gz", "")
        base_id = filename.replace("_t1c", "")  # ⬅️ ensure we match patient_id format in CSV
        split = split_dict.get(base_id, "r_test")  # fallback to r_test if not found
        sub_output = f"/Users/iujeong/0.local/1.bet_all/{split.replace('r_', 'b_')}"
        os.makedirs(sub_output, exist_ok=True)

        bet_output = os.path.join(sub_output, filename + "_bet.nii.gz")
        bet_mask_output = os.path.join(sub_output, filename + "_bet_mask.nii.gz")

        # FastSurfer 수행
        subject_id = filename
        subprocess.run([
            "/Users/iujeong/0.local/FastSurfer/run_fastsurfer.sh",
            "--t1", nii_path,
            "--sid", subject_id,
            "--sd", sub_output,
            "--seg_only", "--no_cuda", "--parallel",
            "--device", device
        ], check=True)

        # BET 마스크는 brainmask.mgz로 생성됨
        img = nib.load(nii_path).get_fdata()
        bet_mask_path = os.path.join(sub_output, subject_id, "mri", "brainmask.mgz")
        bet_mask = nib.load(bet_mask_path).get_fdata()

        # gtv mask 경로 추정
        gtv_path = os.path.join(gtv_dir, filename + "_gtv_mask.nii.gz")
        if os.path.exists(gtv_path):
            gtv_mask = nib.load(gtv_path).get_fdata()
        else:
            log_f.write(f"{filename}: GTV mask not found!\n")
            continue

        # shape 비교 및 로깅
        match_status = (
            "MATCH" if img.shape == bet_mask.shape == gtv_mask.shape else "MISMATCH"
        )
        log_f.write(
            f"{filename} | img: {img.shape}, bet: {bet_mask.shape}, gtv: {gtv_mask.shape} => {match_status}\n"
        )

def fix_affine(target_path, reference_path, output_path=None):
    """
    target_path: affine을 맞추고 싶은 이미지 경로 (예: brainmask)
    reference_path: 기준이 될 이미지 경로 (예: T1c)
    output_path: 저장 경로 (없으면 _fixed 붙여 저장)
    """
    target_img = nib.load(target_path)
    reference_img = nib.load(reference_path)

    # 데이터 가져오기
    data = target_img.get_fdata()

    # 기준 affine과 header 적용
    fixed_img = nib.Nifti1Image(data, affine=reference_img.affine, header=reference_img.header)

    # 저장 경로 설정
    if output_path is None:
        base, ext = os.path.splitext(target_path)
        if ext == ".gz":
            base, ext2 = os.path.splitext(base)
            output_path = base + "_fixed.nii.gz"
        else:
            output_path = base + "_fixed" + ext

    nib.save(fixed_img, output_path)
    print(f"✅ affine 맞춰 저장됨: {output_path}")

import shutil

# patient_ids_by_split.csv 기준으로 환자별 결과 디렉토리 이동
split_csv = "/Users/iujeong/0.local/2.resample/patient_ids_by_split.csv"
df_split = pd.read_csv(split_csv)

src_base = "/Users/iujeong/0.local/1.bet_all"
dst_base = "/Users/iujeong/0.local/1.bet_all_reorganized"

for _, row in df_split.iterrows():
    split = row["split"].replace("r_", "b_")  # r_train -> b_train
    pid = row["patient_id"]
    src_dir = os.path.join(src_base, split, f"{pid}_t1c")
    dst_dir = os.path.join(dst_base, split, pid)

    if os.path.exists(src_dir):
        os.makedirs(os.path.dirname(dst_dir), exist_ok=True)
        shutil.move(src_dir, dst_dir)
        print(f"📁 이동: {src_dir} -> {dst_dir}")
    else:
        print(f"⚠️ 없음: {src_dir}")