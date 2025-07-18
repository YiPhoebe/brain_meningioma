import os
import csv
import nibabel as nib

# 각 split별 경로 설정
splits = {
    "train": {
        "before": "/Users/iujeong/0.local/3.normalize/n_train",
        "after": "/Users/iujeong/0.local/4.slice/s_train/npy"
    },
    "val": {
        "before": "/Users/iujeong/0.local/3.normalize/n_val",
        "after": "/Users/iujeong/0.local/4.slice/s_val/npy"
    },
    "test": {
        "before": "/Users/iujeong/0.local/3.normalize/n_test",
        "after": "/Users/iujeong/0.local/4.slice/s_test/npy"
    }
}

output_csv = "/Users/iujeong/0.local/4.slice/check_code/slice_count_diff.csv"

def count_slices(root, pid, is_npy=True):
    if is_npy:
        return len([f for f in os.listdir(root) if f.startswith(pid) and f.endswith("_img.npy")])
    else:
        gtv_path = os.path.join(root, f"{pid}_gtv_mask.nii.gz")
        if not os.path.exists(gtv_path):
            return 0
        data = nib.load(gtv_path).get_fdata()
        return data.shape[2]

# CSV 저장
with open(output_csv, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["split", "patient_id", "total_slices", "after_filtering", "removed_slices"])

    for split, paths in splits.items():
        before_root = paths["before"]
        after_root = paths["after"]

        before_ids = sorted(set(f.split("_gtv_mask")[0] for f in os.listdir(before_root) if f.endswith("_gtv_mask.nii.gz")))
        after_ids = sorted(set(f.split("_")[0] for f in os.listdir(after_root) if f.endswith(".npy")))
        all_ids = sorted(set(before_ids) | set(after_ids))

        for pid in all_ids:
            total = count_slices(before_root, pid, is_npy=False)
            filtered = count_slices(after_root, pid, is_npy=True)
            removed = total - filtered
            writer.writerow([split, pid, total, filtered, removed])

print(f"✅ 저장 완료: {output_csv}")