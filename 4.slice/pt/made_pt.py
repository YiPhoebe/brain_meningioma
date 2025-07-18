import os
import torch
import numpy as np
from glob import glob
from tqdm import tqdm

def cache_dataset(npy_dir, save_path):
    img_paths = sorted(glob(os.path.join(npy_dir, "*_img.npy")))
    mask_paths = sorted(glob(os.path.join(npy_dir, "*_mask.npy")))

    images = [torch.from_numpy(np.load(p)).unsqueeze(0) for p in tqdm(img_paths, desc="Loading images")]
    masks  = [torch.from_numpy(np.load(p)).unsqueeze(0) for p in tqdm(mask_paths, desc="Loading masks")]

    torch.save((images, masks), save_path)
    print(f"✅ Saved: {save_path}")


cache_dataset("/Users/iujeong/0.local/4.slice/s_train/npy", "/Users/iujeong/0.local/4.slice/pt/train_cache.pt")
cache_dataset("/Users/iujeong/0.local/4.slice/s_val/npy",   "/Users/iujeong/0.local/4.slice/pt/val_cache.pt")
cache_dataset("/Users/iujeong/0.local/4.slice/s_test/npy",  "/Users/iujeong/0.local/4.slice/pt/test_cache.pt")