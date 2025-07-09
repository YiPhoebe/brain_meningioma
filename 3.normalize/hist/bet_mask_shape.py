import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# CSV 로드
df = pd.read_csv('/Users/iujeong/0.local/8.result/csv/bet_bbox_stats.csv')

# 가로, 세로 길이 계산
df['bbox_width'] = df['x_max'] - df['x_min']
df['bbox_height'] = df['y_max'] - df['y_min']

def plot_distribution(data, col, color, title, xlabel):
    plt.figure(figsize=(10, 5))
    sns.histplot(data[col], bins=30, kde=True, color=color, edgecolor='black', alpha=0.3)

    mean_val = data[col].mean()
    p95_val = data[col].quantile(0.95)

    plt.axvline(mean_val, color=color, linestyle='--', linewidth=2, label=f'Mean: {mean_val:.1f}')
    plt.axvline(p95_val, color=color, linestyle=':', linewidth=2, label=f'95%: {p95_val:.1f}')

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel('Count')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'/Users/iujeong/0.local/3.normalize/hist/{col}_distribution.png')
    plt.show()

# Width 그래프 (빨간색)
plot_distribution(df, 'bbox_width', 'red', 'Width Distribution of Slices', 'Width (pixels)')

# Height 그래프 (파란색)
plot_distribution(df, 'bbox_height', 'blue', 'Height Distribution of Slices', 'Height (pixels)')