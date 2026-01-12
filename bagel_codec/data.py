from __future__ import annotations
import os
from typing import List
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T

IMG_EXT = (".png", ".jpg", ".jpeg", ".webp", ".bmp")

class ImageFolderDataset(Dataset):
    def __init__(self, root: str, size: int = 512):
        self.paths: List[str] = []
        for dp, _, fns in os.walk(root):
            for fn in fns:
                if fn.lower().endswith(IMG_EXT):
                    self.paths.append(os.path.join(dp, fn))
        self.paths.sort()
        self.tf = T.Compose([
            T.RandomResizedCrop(size, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
        ])

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        path = self.paths[idx]
        img = Image.open(path).convert("RGB")
        x = self.tf(img)
        return {"image": x, "path": path}
