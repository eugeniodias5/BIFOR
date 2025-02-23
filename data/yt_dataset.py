import os

import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class YtDogDataset(Dataset):
    def __init__(
        self, root_path, transform=None
    ):
        self.root_path = root_path
        self.transform = transform

        self.targets = []
        self.samples = []

        self.positive_sets = None
        self.negative_sets = None

        for target in os.listdir(root_path):
            target_path = os.path.join(root_path, target)
            for sample in os.listdir(target_path):
                # Check if the file is an image
                if sample.lower().endswith((".png", ".jpg", ".jpeg")):
                    self.targets.append(target)
                    self.samples.append(os.path.join(target_path, sample))

        # Convert to numpy array
        self.targets = np.array(self.targets)
        self.samples = np.array(self.samples)

    def __len__(self):
        return len(self.targets)
    
    def __getitem__(self, idx):
        """
        Args:
            idx (int or tuple): An index or a tuple of (anchor_idx, positive_idx, negative_idx).

        Returns:
            Tuple of (anchor_data, positive_data, negative_data).
        """
        if isinstance(idx, tuple):
            # If this is true, we are under offline triplet mining

            anchor_idx, positive_idx, negative_idx = idx
            target = self.targets[anchor_idx]
            anchor_sample = self.samples[anchor_idx]
            positive_sample = self.samples[positive_idx]
            negative_sample = self.samples[negative_idx]

            anchor = Image.open(anchor_sample).convert("RGB")
            positive = Image.open(positive_sample).convert("RGB")
            negative = Image.open(negative_sample).convert("RGB")

            if self.transform:
                anchor = self.transform(anchor)
                positive = self.transform(positive)
                negative = self.transform(negative)

            return anchor, positive, negative, target

        else:
            sample = self.samples[idx]
            target = self.targets[idx]

            anchor = Image.open(sample).convert("RGB")

            if self.transform:
                anchor = self.transform(anchor)
            
            return anchor, target
