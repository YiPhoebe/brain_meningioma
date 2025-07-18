import os
import numpy as np
import pandas as pd
from glob import glob

# 환자별 성능 저장
results = []

# 데이터 루트 설정 (train/val/test 모두 포함)
splits = ["train", "val", "test"]
root_dir = "/Users/iujeong/0.local/4.slice"

def dice_score(y_true, y_pred):
    intersection = np.sum(y_true * y_pred)
    return (2. * intersection) / (np.sum(y_true) + np.sum(y_pred) + 1e-8)

def iou_score(y_true, y_pred):
    intersection = np.sum(y_true * y_pred)
    union = np.sum(y_true) + np.sum(y_pred) - intersection
    return intersection / (union + 1e-8)

for split in splits:
    pred_files = sorted(glob(os.path.join(root_dir, f"s_{split}/npy/*_pred.npy")))

    patients = {}
    for pred_path in pred_files:
        fname = os.path.basename(pred_path)
        pid = fname.split("_slice_")[0]

        # 대응하는 GT 마스크 로딩
        mask_path = pred_path.replace("_pred.npy", "_mask.npy")
        if not os.path.exists(mask_path):
            continue

        pred = np.load(pred_path)
        mask = np.load(mask_path)

        # 이진화
        pred = (pred > 0.5).astype(np.uint8)
        mask = (mask > 0.5).astype(np.uint8)

        d = dice_score(mask, pred)
        i = iou_score(mask, pred)

        if pid not in patients:
            patients[pid] = {"dice": [], "iou": []}
        patients[pid]["dice"].append(d)
        patients[pid]["iou"].append(i)

    # 평균 성능 계산
    for pid, metrics in patients.items():
        dice_mean = np.mean(metrics["dice"])
        iou_mean = np.mean(metrics["iou"])

        # 이유 추론: 0.0인 슬라이스 중 첫 번째 이유 저장
        reason = ""
        if np.isclose(dice_mean, 0.0):
            for pred_path in pred_files:
                if pid not in pred_path:
                    continue
                mask_path = pred_path.replace("_pred.npy", "_mask.npy")
                if not os.path.exists(mask_path):
                    continue
                pred = np.load(pred_path)
                mask = np.load(mask_path)
                pred = (pred > 0.5).astype(np.uint8)
                mask = (mask > 0.5).astype(np.uint8)
                if np.sum(mask) > 0 and np.sum(pred) == 0:
                    reason = "GT positive, pred all zero"
                    break
                elif np.sum(mask) == 0 and np.sum(pred) == 0:
                    reason = "Both GT and pred empty"
                    break
                else:
                    reason = "Other zero Dice case"
                    break

        results.append({
            "patient_id": pid,
            "dice": dice_mean,
            "iou": iou_mean,
            "zero_reason": reason if reason else None
        })

# 저장
save_path = "/Users/iujeong/0.local/6.modelling/result/per_patient_metrics.csv"
df = pd.DataFrame(results)
df.to_csv(save_path, index=False)
print(f"✅ 환자별 성능 저장 완료: {save_path}")

# --- (추가) dice=0.0이고 zero_reason이 'Other zero Dice case'인 환자 예측/GT 시각화 PNG 저장 ---
import matplotlib.pyplot as plt

# 시각화 저장 경로
vis_dir = "/Users/iujeong/0.local/6.modelling/result/zero_case_pngs"
os.makedirs(vis_dir, exist_ok=True)

# 0.0이고 Other zero Dice case인 환자만 필터링
for result in results:
    if not np.isclose(result["dice"], 0.0):
        continue
    if result["zero_reason"] != "Other zero Dice case":
        continue
    pid = result["patient_id"]
    for split in splits:
        pred_files = sorted(glob(os.path.join(root_dir, f"s_{split}/npy/{pid}_slice_*_pred.npy")))
        for pred_path in pred_files:
            slice_id = pred_path.split("_slice_")[1].split("_")[0]
            mask_path = pred_path.replace("_pred.npy", "_mask.npy")
            if not os.path.exists(mask_path):
                continue
            pred = np.load(pred_path)
            mask = np.load(mask_path)

            plt.figure(figsize=(8, 4))
            plt.subplot(1, 2, 1)
            plt.imshow(mask, cmap='gray')
            plt.title('GT Mask')
            plt.axis('off')

            plt.subplot(1, 2, 2)
            plt.imshow(pred, cmap='hot')
            plt.title('Prediction')
            plt.axis('off')

            save_file = os.path.join(vis_dir, f"{pid}_{slice_id}.png")
            plt.tight_layout()
            plt.savefig(save_file)
            plt.close()

# --- (추가) 슬라이스 수 적은 그룹 vs 일반 그룹 평균 비교 및 t-test ---
import scipy.stats as stats

# 슬라이스 수 정보 불러오기
slice_csv = "/Users/iujeong/0.local/4.slice/check_code/slice_count_diff.csv"
df_slice = pd.read_csv(slice_csv)

# 병합
df_metrics = pd.DataFrame(results)
df_merge = pd.merge(df_metrics, df_slice, on="patient_id", how="inner")

# 그룹 나누기: 슬라이스 수 5장 이하 vs 그 외
low_slice = df_merge[df_merge["after_filtering"] <= 5]
normal_slice = df_merge[df_merge["after_filtering"] > 5]

# 평균 Dice
print("🧠 평균 Dice")
print(f"- 5장 이하 ({len(low_slice)}명): {low_slice['dice'].mean():.4f}")
print(f"- 6장 이상 ({len(normal_slice)}명): {normal_slice['dice'].mean():.4f}")

# t-test
t, p = stats.ttest_ind(low_slice["dice"], normal_slice["dice"], equal_var=False)
print("\n📊 t-test 결과 (Welch’s t-test)")
print(f"- t-statistic: {t:.4f}")
print(f"- p-value: {p:.4f}")

# 결과 CSV로 저장
result_df = pd.DataFrame([
    {
        "group": "≤5 slices",
        "count": len(low_slice),
        "mean_dice": round(low_slice["dice"].mean(), 4),
        "t_statistic": round(t, 4),
        "p_value": round(p, 4)
    },
    {
        "group": "≥6 slices",
        "count": len(normal_slice),
        "mean_dice": round(normal_slice["dice"].mean(), 4),
        "t_statistic": "",
        "p_value": ""
    }
])
stat_path = "/Users/iujeong/0.local/6.modelling/result/slice_vs_dice_stats.csv"
result_df.to_csv(stat_path, index=False)

print(f"\n✅ 그룹 비교 통계 저장 완료: {stat_path}")

# --- (추가) 히스토그램 시각화 저장 ---
import matplotlib.pyplot as plt

plt.figure(figsize=(7,4))
plt.hist(df_merge["after_filtering"], bins=range(0, df_merge["after_filtering"].max() + 5, 5), edgecolor='black')
plt.xlabel("Number of Slices (after filtering)")
plt.ylabel("Number of Patients")
plt.title("Distribution of Slice Counts per Patient")
plt.tight_layout()
plt.savefig("/Users/iujeong/0.local/6.modelling/result/slice_count_histogram.png", dpi=300)
print("📊 슬라이스 수 분포 히스토그램 저장 완료")

# --- (추가) 평균 Dice 막대그래프 저장 ---
plt.figure(figsize=(5,4))
plt.bar(result_df["group"], result_df["mean_dice"], color=['salmon', 'skyblue'], edgecolor='black')
plt.ylabel("Mean Dice")
plt.title("Mean Dice by Slice Count Group")
for idx, val in enumerate(result_df["mean_dice"]):
    plt.text(idx, val + 0.01, f"{val:.3f}", ha='center', fontsize=10)
plt.ylim(0, max(result_df["mean_dice"]) + 0.05)
plt.tight_layout()
plt.savefig("/Users/iujeong/0.local/6.modelling/result/mean_dice_barplot.png", dpi=300)
print("📊 평균 Dice 막대그래프 저장 완료")