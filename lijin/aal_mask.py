import nibabel as nib
import numpy as np

# 전체 마스크 로드
mask_img = nib.load("/Users/iujeong/0.local/lijin/AAL3v1.nii.gz")
mask_data = mask_img.get_fdata()
affine = mask_img.affine

region_names = {
    "frontal": list(range(3, 33)) + [15, 16, 17, 18, 19, 20],
    "parietal": [61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74],
    "temporal": list(range(83, 95)),
    "cerebellum": list(range(95, 121)),
    "thalamus": list(range(121, 151)),
}

for name, labels in region_names.items():
    region_mask = np.isin(mask_data, labels).astype(np.uint8)
    print(f"{name} voxel 수:", np.sum(region_mask))
    region_img = nib.Nifti1Image(region_mask, affine)
    nib.save(region_img, f"/Users/iujeong/0.local/lijin/{name}_mask.nii.gz")  # atlas에서 직접 생성