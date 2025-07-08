import os
from pathlib import Path
import torch
import torchio as tio

input_root = Path("/Users/iujeong/0.local/bet_all_old")
output_root = Path("/Users/iujeong/0.local/2.resample/test")

device = torch.device('mps' if torch.cuda.is_available() else 'cpu')
new_spacing = (1.0, 1.0, 1.0)

def resample_volume(input_path, output_path, target_image_path, is_label=False):
    if is_label:
        image = tio.LabelMap(str(input_path))
    else:
        image = tio.ScalarImage(str(input_path))
    target = tio.ScalarImage(str(target_image_path))
    resample_transform = tio.Resample(target=target, image_interpolation='nearest' if is_label else 'linear')
    resampled = resample_transform(image)

    import numpy as np
    data = np.asanyarray(resampled.data.data)
    if is_label:
        data = data.astype(np.uint8)
    tensor = torch.from_numpy(data)
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)
    resampled.set_data(tensor)

    resampled.save(str(output_path))
    print(f"✅ Resampled: {input_path.name} → {output_path}")

split_dirs = ["b_test", "b_train", "b_val"]

# Only process specific cases with mismatched shapes
selected_cases = [
    "BraTS-MEN-RT-0154-1",
    "BraTS-MEN-RT-0369-1",
    "BraTS-MEN-RT-0455-1",
]

output_dir = output_root
output_dir.mkdir(parents=True, exist_ok=True)

for case_id in selected_cases:
    found = False
    for split in split_dirs:
        input_dir = input_root / split
        t1c_path = input_dir / f"{case_id}_t1c_bet.nii.gz"
        gtv_path = input_dir / f"{case_id}_t1c_gtv_mask.nii.gz"
        bet_mask_path = input_dir / f"{case_id}_t1c_bet_mask.nii.gz"

        if t1c_path.exists() and gtv_path.exists() and bet_mask_path.exists():
            resample_volume(t1c_path, output_dir / t1c_path.name, t1c_path, is_label=False)
            resample_volume(gtv_path, output_dir / gtv_path.name, t1c_path, is_label=True)
            resample_volume(bet_mask_path, output_dir / bet_mask_path.name, t1c_path, is_label=True)
            found = True
            break

    if not found:
        print(f"❌ Files for {case_id} not found in any split folder.")

print("\n📏 Verifying shapes match for each case...\n")
log_path = output_root / "resample_shape_mismatch.log"
with open(log_path, "w") as log_file:
    for case_id in selected_cases:
        t1c_path = output_dir / f"{case_id}_t1c_bet.nii.gz"
        bet_path = output_dir / f"{case_id}_t1c_bet_mask.nii.gz"
        gtv_path = output_dir / f"{case_id}_t1c_gtv_mask.nii.gz"

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