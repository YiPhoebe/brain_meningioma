import os
from pathlib import Path
import torch
import torchio as tio

input_t1c_dir = Path("/home/iujeong/brain_meningioma/raw_data/all_t1c")
input_gtv_dir = Path("/home/iujeong/brain_meningioma/raw_data/all_gtv")
output_t1c_dir = Path("/home/iujeong/brain_meningioma/raw_data/resampled/all_t1c")
output_gtv_dir = Path("/home/iujeong/brain_meningioma/raw_data/resampled/all_gtv")

output_t1c_dir.mkdir(parents=True, exist_ok=True)
output_gtv_dir.mkdir(parents=True, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
new_spacing = (1.0, 1.0, 1.0)

def resample_t1c(image_path, output_path, new_spacing=new_spacing):
    image = tio.ScalarImage(str(image_path))
    resample_transform = tio.Resample(new_spacing, image_interpolation='bspline')
    resampled = resample_transform(image)
    resampled.save(str(output_path))
    print(f"✅ Resampled T1c: {image_path.name} → {output_path}")

def resample_gtv(image_path, output_path, new_spacing=new_spacing):
    label = tio.LabelMap(str(image_path))
    resample_transform = tio.Resample(new_spacing, image_interpolation='nearest')
    resampled = resample_transform(label)
    resampled.save(str(output_path))
    print(f"✅ Resampled GTV: {image_path.name} → {output_path}")

# T1c 리샘플링
for file in sorted(input_t1c_dir.glob("*.nii.gz")):
    out_file = output_t1c_dir / file.name
    resample_t1c(file, out_file)

# GTV 리샘플링 (binary mask → NearestNeighbor)
for file in sorted(input_gtv_dir.glob("*.nii.gz")):
    out_file = output_gtv_dir / file.name
    resample_gtv(file, out_file)