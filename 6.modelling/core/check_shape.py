import numpy as np
import os

root = "/Users/iujeong/0.local/4.slice/s_train/npy"
log_path = "shape_mismatch_log.txt"
log_file = open(log_path, "w")
count = 0

for f in sorted(os.listdir(root)):
    if not f.endswith("_img.npy"):
        continue

    img_path = os.path.join(root, f)
    mask_path = img_path.replace("_img.npy", "_mask.npy")

    if not os.path.exists(mask_path):
        log_file.write(f"{f} → ❌ 마스크 없음\n")
        print(f"{f} → ❌ 마스크 없음")
        continue

    img = np.load(img_path)
    mask = np.load(mask_path)

    if img.shape != mask.shape:
        log_file.write(f"{f} → 🚨 shape mismatch: image={img.shape}, mask={mask.shape}\n")
        print(f"{f} → 🚨 shape mismatch: image={img.shape}, mask={mask.shape}")
        count += 1

log_file.close()
with open(log_path, "a") as log_file:
    log_file.write(f"\n총 mismatch 개수: {count}\n")