import os
import nibabel as nib
from glob import glob
from collections import defaultdict

folders = [
    '/Users/iujeong/0.local/2.resample/r_train',
    '/Users/iujeong/0.local/2.resample/r_val',
    '/Users/iujeong/0.local/2.resample/r_test',
]

nii_files = []
for folder in folders:
    nii_files.extend(sorted(glob(os.path.join(folder, '*.nii.gz'))))

# 그룹핑: key = 환자 ID
groups = defaultdict(dict)
for path in nii_files:
    fname = os.path.basename(path)
    if '_gtv_mask' in fname:
        key = fname.replace('_t1c_gtv_mask.nii.gz', '')
        groups[key]['gtv_mask'] = path
    elif '_bet_mask' in fname:
        key = fname.replace('_t1c_bet_mask.nii.gz', '')
        groups[key]['bet_mask'] = path
    elif '_bet.nii.gz' in fname:
        key = fname.replace('_t1c_bet.nii.gz', '')
        groups[key]['bet'] = path

# 검사
for pid, files in groups.items():
    print(f'\n🧪 {pid}')
    for tag in ['bet', 'bet_mask', 'gtv_mask']:
        path = files.get(tag)
        if not path:
            print(f'⚠️ {tag} 파일 없음')
            continue
        spacing = nib.load(path).header.get_zooms()
        if spacing != (1.0, 1.0, 1.0):
            print(f'❌ {tag}: {spacing}')
        else:
            print(f'✅ {tag}: {spacing}')

import csv

csv_path = '/Users/iujeong/0.local/2.resample/spacing_check.csv'
with open(csv_path, mode='w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['patient_id', 'file_type', 'spacing_x', 'spacing_y', 'spacing_z', 'status'])
    for pid, files in groups.items():
        for tag in ['bet', 'bet_mask', 'gtv_mask']:
            path = files.get(tag)
            if not path:
                continue
            spacing = nib.load(path).header.get_zooms()
            status = 'ok' if spacing == (1.0, 1.0, 1.0) else 'not_ok'
            writer.writerow([pid, tag, spacing[0], spacing[1], spacing[2], status])