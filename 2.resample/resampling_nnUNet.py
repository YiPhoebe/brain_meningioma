import os
from pathlib import Path
import torch
import torchio as tio
import nibabel as nib
import numpy as np
from nnunetv2.preprocessing.resampling.resample_torch import resample_torch_fornnunet

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
print(device)

input_root = Path("/Users/iujeong/0.local/1.bet_all")
output_root = Path("/Users/iujeong/0.local/resampling")

def resample_volume(input_path, output_path, target_image_path):
    is_mask = any(tag in input_path.name for tag in ["_bet_mask", "_gtv"])
    nii = nib.load(str(input_path))
    data = nii.get_fdata()
    spacing = tuple(float(s) for s in nii.header.get_zooms())
    shape = tuple(int(s) for s in data.shape)
    tensor = torch.from_numpy(data).unsqueeze(0)
    tensor = tensor.to(torch.float32) if not is_mask else tensor.to(torch.uint8)

    new_spacing = tuple(float(s) for s in (1.0, 1.0, 1.0))
    new_shape = [
        int(round(float(s) / float(ns) * float(sh)))
        for s, ns, sh in zip(spacing, new_spacing, shape)
    ]
    new_shape = tuple(int(v) for v in new_shape)

    resampled = resample_torch_fornnunet(
        data=tensor,
        new_shape=new_shape,
        current_spacing=spacing,
        new_spacing=new_spacing,
        is_seg=is_mask,
        device=device,
        force_separate_z=True,
    )

    npy = resampled.squeeze(0).cpu().numpy()
    if is_mask:
        npy = npy.astype(np.uint8)
    else:
        npy = npy.astype(np.float32)

    new_nii = nib.Nifti1Image(npy, affine=nii.affine)
    nib.save(new_nii, str(output_path))
    print(f"✅ Resampled: {input_path.name} → {output_path}")

split_dirs = ["b_test", "b_train", "b_val"]

selected_cases = []

for split in split_dirs:
    output_dir = output_root / split.replace("b_", "r_")
    output_dir.mkdir(parents=True, exist_ok=True)

    input_dir = input_root / split
    for t1c_path in sorted(input_dir.glob("*_t1c_bet.nii.gz")):
        case_id = t1c_path.name.replace("_t1c_bet.nii.gz", "")
        gtv_path = input_dir / f"{case_id}_t1c_gtv_mask.nii.gz"
        bet_mask_path = input_dir / f"{case_id}_t1c_bet_mask.nii.gz"

        if t1c_path.exists() and gtv_path.exists() and bet_mask_path.exists():
            resample_volume(t1c_path, output_dir / f"{case_id}_t1c_bet.nii.gz", t1c_path)
            resample_volume(gtv_path, output_dir / f"{case_id}_t1c_gtv_mask.nii.gz", gtv_path)
            resample_volume(bet_mask_path, output_dir / f"{case_id}_t1c_bet_mask.nii.gz", t1c_path)
            selected_cases.append(case_id)
        else:
            print(f"❌ Missing file(s) for {case_id} in {split}")

print("\n📏 Verifying shapes match for each case...\n")
log_path = Path("/Users/iujeong/0.local/resampling") / "resample_shape_mismatch.log"
with open(log_path, "w") as log_file:
    for split in split_dirs:
        phase = split.replace("b_", "r_")
        output_dir = output_root / phase
        for case_id in selected_cases:
            t1c_path = output_dir / f"{case_id}_t1c_bet.nii.gz"
            bet_path = output_dir / f"{case_id}_t1c_bet_mask.nii.gz"
            gtv_path = output_dir / f"{case_id}_t1c_gtv_mask.nii.gz"

            if not (t1c_path.exists() and bet_path.exists() and gtv_path.exists()):
                continue

            try:
                t1c = tio.ScalarImage(str(t1c_path))
                bet = tio.LabelMap(str(bet_path))
                gtv = tio.LabelMap(str(gtv_path))

                t1c_shape = tuple(t1c.shape)
                bet_shape = tuple(bet.shape)
                gtv_shape = tuple(gtv.shape)

                if t1c_shape != bet_shape or t1c_shape != gtv_shape:
                    msg = (
                        f"[❌] Shape mismatch: {case_id}\n"
                        f"    T1c: {t1c_shape}, BET: {bet_shape}, GTV: {gtv_shape}\n"
                    )
                    print(msg)
                    log_file.write(msg)
            except Exception as e:
                err_msg = f"[⚠️] Failed to read {case_id}: {e}\n"
                print(err_msg)
                log_file.write(err_msg)
print(f"\n🔍 Shape mismatch log saved to: {log_path}")
