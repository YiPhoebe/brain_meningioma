

import os
from pathlib import Path
import torch
import torchio as tio

input_root = Path("/Users/iujeong/0.local/1.bet_all")
output_root = Path("/Users/iujeong/0.local/2.resample")

device = torch.device('mps' if torch.cuda.is_available() else 'cpu')
new_spacing = (1.0, 1.0, 1.0)

def resample_volume(input_path, output_path, target_image_path):
    # Automatically determine if this is a mask/label based on filename
    is_mask = any(tag in input_path.name for tag in ["_bet_mask", "_gtv"])
    if is_mask:
        image = tio.LabelMap(str(input_path))
    else:
        image = tio.ScalarImage(str(input_path))
    reference = tio.ScalarImage(str(target_image_path))
    resample_transform = tio.Resample(reference, image_interpolation='nearest' if is_mask else 'linear')
    resampled = resample_transform(image)

    import numpy as np
    data = np.asanyarray(resampled.data.data)
    if is_mask:
        data = np.round(data).astype(np.uint8)
    else:
        data = data.astype(np.float32)
    tensor = torch.from_numpy(data)
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)
    resampled.set_data(tensor)

    resampled.save(str(output_path))
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
            # 1. Load and resample T1c image to (1, 1, 1) spacing
            t1c_img = tio.ScalarImage(str(t1c_path))
            t1c_resampled = tio.Resample((1, 1, 1))(t1c_img)
            t1c_resampled_path = output_dir / f"{case_id}_t1c_bet.nii.gz"
            t1c_resampled.save(str(t1c_resampled_path))

            # 2. Use resampled T1c as reference for GTV and BET masks
            resample_volume(gtv_path, output_dir / f"{case_id}_t1c_gtv_mask.nii.gz", t1c_resampled_path)
            resample_volume(bet_mask_path, output_dir / f"{case_id}_t1c_bet_mask.nii.gz", t1c_resampled_path)
            selected_cases.append(case_id)
        else:
            print(f"❌ Missing file(s) for {case_id} in {split}")

print("\n📏 Verifying shapes match for each case...\n")
log_path = Path("/Users/iujeong/0.local/2.resample") / "resample_shape_mismatch.log"
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