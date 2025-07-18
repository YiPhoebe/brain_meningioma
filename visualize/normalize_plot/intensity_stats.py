import os
import glob
import nibabel as nib
import numpy as np
import pandas as pd
from tqdm import tqdm

# 경로 설정
resample_root = '/Users/iujeong/0.local/2.resample'
normalize_root = '/Users/iujeong/0.local/3.normalize'
sets = ['r_train', 'r_val', 'r_test']
csv_save_path = '/Users/iujeong/0.local/visualize/normalize_plot/intensity_stats.csv'

records = []

for subset in sets:
    resample_dir = os.path.join(resample_root, subset)
    normalize_dir = os.path.join(normalize_root, 'n' + subset[1:])  # r_train → n_train

    nii_paths = sorted(glob.glob(os.path.join(resample_dir, '*_t1c_bet.nii.gz')))
    for resample_path in tqdm(nii_paths):
        fname = os.path.basename(resample_path).replace('_t1c_bet.nii.gz', '')
        norm_path = os.path.join(normalize_dir, f'{fname}_norm.nii.gz')
        norm_bet_mask_fname = fname.replace('_t1c', '')  # 정규화 후에는 t1c 빠짐
        bet_mask_path = os.path.join(normalize_dir, f'{norm_bet_mask_fname}_bet_mask.nii.gz')

        if not os.path.exists(norm_path) or not os.path.exists(bet_mask_path):
            print(f"Skip: {fname}")
            if not os.path.exists(norm_path):
                print(f"  ⛔ norm image missing: {norm_path}")
            if not os.path.exists(bet_mask_path):
                print(f"  ⛔ bet mask missing: {bet_mask_path}")
            continue

        try:
            orig_img = nib.load(resample_path).get_fdata()
            norm_img = nib.load(norm_path).get_fdata()
            bet_mask = nib.load(bet_mask_path).get_fdata()
        except:
            print(f"Error loading: {fname}")
            continue

        brain_mask = bet_mask > 0
        orig_vals = orig_img[brain_mask]
        norm_vals = norm_img[brain_mask]

        records.append({
            'ID': fname,
            'Subset': subset,
            'Orig_Mean': np.mean(orig_vals),
            'Orig_Std': np.std(orig_vals),
            'Orig_Min': np.min(orig_vals),
            'Orig_Max': np.max(orig_vals),
            'Norm_Mean': np.mean(norm_vals),
            'Norm_Std': np.std(norm_vals),
            'Norm_Min': np.min(norm_vals),
            'Norm_Max': np.max(norm_vals)
        })

# 저장
df = pd.DataFrame(records)
df.to_csv(csv_save_path, index=False)
print(f"Saved to: {csv_save_path}")

# 로그 저장
log_path = csv_save_path.replace('.csv', '.log')
with open(log_path, 'w') as f:
    f.write("✅ Intensity Statistics Log (BET 영역 기준)\n")

    if df.empty:
        f.write("⚠️ 데이터가 없습니다. 모든 샘플이 누락되었거나 로딩에 실패했습니다.\n")
    else:
        f.write(f"• 총 처리 환자 수: {len(df)}명\n")

        for subset_name in df['Subset'].unique():
            count = (df['Subset'] == subset_name).sum()
            f.write(f"• {subset_name}: {count}명\n")

        f.write("\n• 전체 평균 (정규화 전): {:.2f} ± {:.2f}\n".format(df['Orig_Mean'].mean(), df['Orig_Std'].mean()))
        f.write("• 전체 평균 (정규화 후): {:.2f} ± {:.2f}\n".format(df['Norm_Mean'].mean(), df['Norm_Std'].mean()))
        f.write("• intensity 범위 (정규화 전): {:.2f} ~ {:.2f}\n".format(df['Orig_Min'].min(), df['Orig_Max'].max()))
        f.write("• intensity 범위 (정규화 후): {:.2f} ~ {:.2f}\n".format(df['Norm_Min'].min(), df['Norm_Max'].max()))

    f.write(f"\n📍 저장된 csv 경로: {csv_save_path}\n")