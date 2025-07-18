from glob import glob
import os

for phase in ["s_test", "s_train", "s_val"]:
    path = f"/Users/iujeong/0.local/4.slice/{phase}/npy"
    files = sorted(glob(os.path.join(path, "*_img.npy"), recursive=True))

    if not files:
        print(f"[WARN] {phase}에 해당하는 파일 없음! 경로 확인 필요: {path}")
    else:
        patient_ids = set(os.path.basename(f).split("_slice_")[0] for f in files)
        print(f"{phase}: {len(patient_ids)}명 | 총 슬라이스: {len(files)}")