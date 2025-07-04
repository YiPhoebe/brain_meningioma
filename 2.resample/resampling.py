

import os
from pathlib import Path
import torch
import torchio as tio

input_root = Path("/Users/iujeong/0.local/1.bet_all")
output_root = Path("/Users/iujeong/0.local/2.resample")

device = torch.device('mps' if torch.cuda.is_available() else 'cpu')
new_spacing = (1.0, 1.0, 1.0)

def resample_t1c(image_path, output_path, new_spacing=new_spacing):
    image = tio.ScalarImage(str(image_path))
    original_min = image.data.min()
    original_max = image.data.max()

    print(f"Before: {original_min.item():.3f} ~ {original_max.item():.3f}")

    resample_transform = tio.Resample(new_spacing, image_interpolation='linear')
    resampled = resample_transform(image)

    print(f"After : {resampled.data.min().item():.3f} ~ {resampled.data.max().item():.3f}")

    # Rescale back to original intensity range
    data = resampled.data
    data = (data - data.min()) / (data.max() - data.min())  # Normalize to [0, 1]
    data = data * (original_max - original_min) + original_min  # Scale to original range
    resampled.data = data

    resampled.save(str(output_path))
    print(f"✅ Resampled T1c: {image_path.name} → {output_path}")

def resample_gtv(image_path, output_path, new_spacing=new_spacing):
    label = tio.LabelMap(str(image_path))
    resample_transform = tio.Resample(new_spacing, image_interpolation='nearest')
    resampled = resample_transform(label)
    resampled.save(str(output_path))
    print(f"✅ Resampled GTV: {image_path.name} → {output_path}")

splits = {
    "b_test": "r_test",
    "b_train": "r_train",
    "b_val": "r_val",
}

for input_split, output_split in splits.items():
    input_dir = input_root / input_split
    output_dir = output_root / output_split
    output_dir.mkdir(parents=True, exist_ok=True)

    # T1c resampling
    for file in sorted(input_dir.glob("*_bet.nii.gz")):
        out_file = output_dir / file.name
        resample_t1c(file, out_file)

    # GTV resampling
    for file in sorted(input_dir.glob("*_t1c_gtv_mask.nii.gz")):
        out_file = output_dir / file.name
        resample_gtv(file, out_file)

    # BET mask resampling
    for file in sorted(input_dir.glob("*_bet_mask.nii.gz")):
        out_file = output_dir / file.name
        resample_gtv(file, out_file)
