import os
import torch
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm
from collections import defaultdict
from medpy.metric.binary import dc, hd95

# ⚙️ CONFIG
PRED_ROOTS = {
    "lg2unetr": "/Users/iujeong/0.local/pred/lg2unetr",
    "swinunetr": "/Users/iujeong/0.local/pred/swinunetr",
    "unet1s": "/Users/iujeong/0.local/pred/unet1s",
    "unetpp0": "/Users/iujeong/0.local/pred/unetpp0",
    "unetpp0D": "/Users/iujeong/0.local/pred/unetpp0D"
}
LABEL_ROOT = "/Users/iujeong/0.local/labels"  # ground truth 폴더 위치 수정 필요

def load_mask(path):
    return torch.load(path).squeeze().numpy()

def compute_metrics(pred, gt):
    try:
        return {
            "Dice": dc(pred, gt),
            "HD95": hd95(pred, gt),
        }
    except:
        return {
            "Dice": 0.0,
            "HD95": np.nan,
        }

def get_centroid(mask):
    coords = np.argwhere(mask > 0)
    if len(coords) == 0:
        return None
    centroid = coords.mean(axis=0)
    return centroid

def assign_region(centroid):
    if centroid is None:
        return np.nan
    # Assuming centroid is (row, col)
    row, col = centroid
    if col < 64:
        return "Left"
    elif col > 192:
        return "Right"
    else:
        return "Center"

def analyze_model(model_name, model_root):
    results = []
    for split in ["s_test", "s_val"]:
        pred_dir = os.path.join(model_root, split, "pt")
        pred_paths = sorted(glob(os.path.join(pred_dir, "*.pt")))

        for pred_path in tqdm(pred_paths, desc=f"{model_name} - {split}"):
            fname = os.path.basename(pred_path).replace("_mask.pt", "")
            gt_path = os.path.join(LABEL_ROOT, split, "label", f"{fname}_label.pt")
            if not os.path.exists(gt_path):
                continue

            pred = load_mask(pred_path)
            gt = load_mask(gt_path)
            metrics = compute_metrics(pred, gt)

            centroid = get_centroid(pred)
            region = assign_region(centroid)

            metrics.update({
                "model": model_name,
                "split": split,
                "patient_id": fname.split("_slice_")[0],
                "slice_id": fname.split("_slice_")[1],
                "gt_sum": int(gt.sum()),  # 종양 크기
                "Region": region,
            })
            results.append(metrics)
    return results

if __name__ == "__main__":
    all_results = []
    for model_name, model_root in PRED_ROOTS.items():
        all_results.extend(analyze_model(model_name, model_root))

    df = pd.DataFrame(all_results)

    df.to_csv("analyze_performance.csv", index=False)
    print("✅ Saved: analyze_performance.csv")