from glob import glob
import os

# 파일 누락 체크
for split in ["train", "val", "test"]:
    base = f"/Users/iujeong/0.local/2.resample/r_{split}"
    bet_imgs = sorted(glob(os.path.join(base, "*_t1c_bet.nii.gz")))
    missing_mask = []
    missing_bet_mask = []

    for img_path in bet_imgs:
        pid = os.path.basename(img_path).split("_t1c")[0]
        mask_path = os.path.join(base, f"{pid}_t1c_gtv_mask.nii.gz")
        bet_mask_path = os.path.join(base, f"{pid}_t1c_bet_mask.nii.gz")
        if not os.path.exists(mask_path):
            missing_mask.append(pid)
        if not os.path.exists(bet_mask_path):
            missing_bet_mask.append(pid)

    print(f"🔍 r_{split} - GTV 마스크 없음: {missing_mask}")
    print(f"🔍 r_{split} - BET 마스크 없음: {missing_bet_mask}")