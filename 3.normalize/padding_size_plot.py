import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# CSV 파일 경로 (수정해서 사용)
csv_dir = "/Users/iujeong/0.local/8.result/csv/slice_shape_stats.csv"

# 저장 경로 폴더 확인
output_dir = "/Users/iujeong/0.local/3.normalize/hist"
os.makedirs(output_dir, exist_ok=True)

# 데이터 불러오기
df = pd.read_csv(csv_dir)

# Height 클리핑 (99% 이하)
height_clip = df[df["height"] <= df["height"].quantile(0.99)]

# Height 분포
plt.figure(figsize=(10, 5))
sns.histplot(height_clip["height"], bins=20, kde=True, color='skyblue')
plt.title("Height Distribution of Slices")
plt.xlabel("Height (pixels)")
plt.ylabel("Count")
plt.grid(True)
height_mean = height_clip["height"].mean()
height_95 = height_clip["height"].quantile(0.95)
height_mode = height_clip["height"].mode()[0]
plt.axvline(height_mean, color='blue', linestyle='--', label=f"Mean: {height_mean:.1f}")
plt.axvline(height_95, color='navy', linestyle=':', label=f"95%: {height_95:.1f}")
plt.axvline(height_mode, color='green', linestyle='-.', label=f"Mode: {height_mode}")
plt.xticks(range(120, 400, 10))
plt.legend()
plt.savefig(os.path.join(output_dir, "height_distribution.png"))
plt.close()

# Width 클리핑 (99% 이하)
width_clip = df[df["width"] <= df["width"].quantile(0.99)]

# Width 분포
plt.figure(figsize=(10, 5))
sns.histplot(width_clip["width"], bins=20, kde=True, color='salmon')
plt.title("Width Distribution of Slices")
plt.xlabel("Width (pixels)")
plt.ylabel("Count")
plt.grid(True)
width_mean = width_clip["width"].mean()
width_95 = width_clip["width"].quantile(0.95)
width_mode = width_clip["width"].mode()[0]
plt.axvline(width_mean, color='red', linestyle='--', label=f"Mean: {width_mean:.1f}")
plt.axvline(width_95, color='darkred', linestyle=':', label=f"95%: {width_95:.1f}")
plt.axvline(width_mode, color='green', linestyle='-.', label=f"Mode: {width_mode}")
plt.xticks(range(120, 400, 10))
plt.legend()
plt.savefig(os.path.join(output_dir, "width_distribution.png"))
plt.close()