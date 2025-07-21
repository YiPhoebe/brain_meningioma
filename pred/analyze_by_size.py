

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load performance data
df = pd.read_csv("per_slice_metrics.csv")

# Drop zero-gt slices to focus on relevant tumor regions
df = df[df["gt_sum"] > 0].copy()

# Assign tumor size group by 3-quantile split
df["size_group"] = pd.qcut(df["gt_sum"], q=3, labels=["S", "M", "L"])

# Grouped performance summary
grouped = df.groupby(["model", "size_group"])["Dice"].agg(["mean", "std", "count"]).reset_index()
print(grouped)

# Save summary table
grouped.to_csv("analyze_by_size.csv", index=False)

# Optional: Boxplot visualization
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x="size_group", y="Dice", hue="model")
plt.title("Dice Score by Tumor Size Group (S/M/L)")
plt.tight_layout()
plt.savefig("analyze_by_size.png")
print("✅ Saved: analyze_by_size.csv, analyze_by_size.png")