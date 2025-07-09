import os
import numpy as np
import torch
from torch.utils.data import Dataset

class MeningiomaDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.files = sorted([f for f in os.listdir(data_dir) if f.endswith('_img.npy')])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.data_dir, self.files[idx])
        mask_path = img_path.replace("_img.npy", "_mask.npy")

        image = np.load(img_path)  # shape: [H, W]
        mask = np.load(mask_path)  # shape: [H, W]

        # Normalize to [0, 1]
        image = image.astype(np.float32)
        image = (image - image.min()) / (image.max() - image.min() + 1e-8)

        image = np.expand_dims(image, axis=0)  # [1, H, W]
        mask = np.expand_dims(mask, axis=0)    # [1, H, W]

        # Debug checks
        if image is None or mask is None:
            raise ValueError(f"[ERROR] {self.files[idx]} → image or mask is None")

        if image.shape != (1, 256, 256):
            raise ValueError(f"[ERROR] {self.files[idx]} → image shape mismatch: {image.shape}")
        if mask.shape != (1, 256, 256):
            raise ValueError(f"[ERROR] {self.files[idx]} → mask shape mismatch: {mask.shape}")

        return torch.tensor(image, dtype=torch.float32), torch.tensor(mask, dtype=torch.float32)