import os
import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio
from glob import glob
from tqdm import tqdm

def main():
    vis_dir = "/Users/iujeong/0.local/5.npy_visualize"
    os.makedirs(vis_dir, exist_ok=True)

    input_dirs = [
        "/Users/iujeong/0.local/4.slice/s_train",
        "/Users/iujeong/0.local/4.slice/s_val",
        "/Users/iujeong/0.local/4.slice/s_test",
    ]

    output_dirs = {
        "s_train": "/Users/iujeong/0.local/5.npy_visualize/o_train",
        "s_val": "/Users/iujeong/0.local/5.npy_visualize/o_val",
        "s_test": "/Users/iujeong/0.local/5.npy_visualize/o_test",
    }

    for input_dir in input_dirs:
        img_paths = sorted(glob(os.path.join(input_dir, "*", "*_img.npy")))
        for img_path in tqdm(img_paths, desc=f"Visualizing {os.path.basename(input_dir)}"):
            img = np.load(img_path)

            print(f"👉 {os.path.basename(img_path)}: min={img.min():.3f}, max={img.max():.3f}, shape={img.shape}")

            base = os.path.basename(img_path).replace("_img.npy", "")
            subdir = os.path.basename(input_dir)
            save_root = output_dirs[subdir]
            os.makedirs(save_root, exist_ok=True)
            save_path = os.path.join(save_root, f"{base}.png")
            plt.imsave(save_path, img, cmap='gray')

if __name__ == "__main__":
    main()