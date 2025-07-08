import nibabel as nib
import os

input_path = "/home/iujeong/brain_meningioma/tmp_input/train/BraTS-MEN-RT-0001-1_t1c.nii.gz"
bet_mask_path = "/home/iujeong/brain_meningioma/0.local/b_train/BraTS-MEN-RT-0001-1_t1c_bet_mask.nii.gz"

img_shape = nib.load(input_path).shape
bet_shape = nib.load(bet_mask_path).shape

print("✅ 일치" if img_shape == bet_shape else f"❌ 불일치: {img_shape} vs {bet_shape}")