import os
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
from PIL import Image

class MeningiomaDataset(Dataset):
    def __init__(self, data_dir, mode="train"):
        self.data_dir = data_dir
        self.mode = mode
        self.files = sorted([f for f in os.listdir(data_dir) if f.endswith('_img.npy')])
        self.image_paths = [os.path.join(self.data_dir, f) for f in self.files]
        self.mask_paths = [img_path.replace("_img.npy", "_mask.npy") for img_path in self.image_paths]

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

        if image.shape != (1, 160, 192):
            raise ValueError(f"[ERROR] {self.files[idx]} → image shape mismatch: {image.shape}")
        if mask.shape != (1, 160, 192):
            raise ValueError(f"[ERROR] {self.files[idx]} → mask shape mismatch: {mask.shape}")

        return torch.tensor(image, dtype=torch.float32), torch.tensor(mask, dtype=torch.float32)


class AugmentedMeningiomaDataset(Dataset):
    def __init__(self, data_dir, mode="train"):
        self.data_dir = data_dir
        self.mode = mode
        self.files = sorted([f for f in os.listdir(data_dir) if f.endswith('_img.npy')])
        self.image_paths = [os.path.join(self.data_dir, f) for f in self.files]
        self.mask_paths = [img_path.replace("_img.npy", "_mask.npy") for img_path in self.image_paths]

        self.transforms = T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.RandomRotation(degrees=15),
            # T.RandomResizedCrop((160, 192), scale=(0.9, 1.0)),
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.data_dir, self.files[idx])
        mask_path = img_path.replace("_img.npy", "_mask.npy")

        image = np.load(img_path)  # shape: [H, W]
        mask = np.load(mask_path)  # shape: [H, W]

        # Convert to PIL Image
        image_pil = Image.fromarray(image.astype(np.uint8))
        mask_pil = Image.fromarray((mask * 255).astype(np.uint8))

        # Apply transforms
        seed = np.random.randint(2147483647)  # make sure image and mask get same transform
        torch.manual_seed(seed)
        image_pil = self.transforms(image_pil)
        torch.manual_seed(seed)
        mask_pil = self.transforms(mask_pil)

        # Convert back to numpy
        image = np.array(image_pil).astype(np.float32)
        mask = np.array(mask_pil).astype(np.float32)
        mask = (mask > 127).astype(np.float32)

        # Normalize image to [0, 1]
        image = (image - image.min()) / (image.max() - image.min() + 1e-8)

        image = np.expand_dims(image, axis=0)  # [1, H, W]
        mask = np.expand_dims(mask, axis=0)    # [1, H, W]

        # Debug checks
        if image is None or mask is None:
            raise ValueError(f"[ERROR] {self.files[idx]} → image or mask is None")

        if image.shape != (1, 160, 192):
            raise ValueError(f"[ERROR] {self.files[idx]} → image shape mismatch: {image.shape}")
        if mask.shape != (1, 160, 192):
            raise ValueError(f"[ERROR] {self.files[idx]} → mask shape mismatch: {mask.shape}")

        return torch.tensor(image, dtype=torch.float32), torch.tensor(mask, dtype=torch.float32)