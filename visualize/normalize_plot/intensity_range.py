import pandas as pd

csv_path = '/Users/iujeong/0.local/visualize/normalize_plot/intensity_stats.csv'
df = pd.read_csv(csv_path)

overall_min = df['Norm_Mean'].min() - df['Norm_Std'].max()
overall_max = df['Norm_Mean'].max() + df['Norm_Std'].max()

print(f"✅ 정규화 후 intensity 범위 (예상): {overall_min:.2f} ~ {overall_max:.2f}")

# 결과 저장
with open('/Users/iujeong/0.local/visualize/normalize_plot/intensity_range.log', 'w') as f:
    f.write(f"정규화 후 intensity 범위 (예상): {overall_min:.2f} ~ {overall_max:.2f}\n")