import os
import pandas as pd
from sklearn.model_selection import train_test_split
from multiprocessing import Pool, cpu_count

# 입력 경로: 원본 T1c 이미지 전체 환자 폴더
input_dir = "/Users/iujeong/0.local/original_data/all_t1c"

# 출력 경로: HD-BET 결과 저장
output_dirs = {
    "train": "/Users/iujeong/0.local/1.bet_all/b_train",
    "test": "/Users/iujeong/0.local/1.bet_all/b_test",
    "val": "/Users/iujeong/0.local/1.bet_all/b_val",
}

test_df = pd.read_csv("/Users/iujeong/0.local/test_files.csv")
test_ids = set(test_df["filename"].str.replace("_t1c.nii.gz", "", regex=False))

all_files = [f for f in os.listdir(input_dir) if f.endswith(".nii.gz")]
test_files = [f for f in all_files if f.replace("_t1c.nii.gz", "") in test_ids]
train_val_files = [f for f in all_files if f not in test_files]

train_files, val_files = train_test_split(train_val_files, test_size=0.1, random_state=42)

split_files = {
    "train": train_files,
    "val": val_files,
    "test": test_files,
}

def run_hd_bet(args):
    input_path, output_path = args
    if os.path.exists(output_path):
        print(f"Skipping {os.path.basename(input_path)} (already processed)")
        return
    cmd = f"/opt/anaconda3/envs/brain_meningioma/bin/hd-bet -i {input_path} -o {output_path} -device cpu"
    print(f"Running HD-BET on {input_path}")
    os.system(cmd)

if __name__ == "__main__":
    tasks = []
    for split in ["train", "val", "test"]:
        output_dir = output_dirs[split]
        os.makedirs(output_dir, exist_ok=True)

        for f in split_files[split]:
            input_path = os.path.join(input_dir, f)
            output_path = os.path.join(output_dir, f.replace(".nii.gz", "_bet.nii.gz"))
            tasks.append((input_path, output_path))

    with Pool(processes=min(8, cpu_count())) as pool:
        pool.map(run_hd_bet, tasks)