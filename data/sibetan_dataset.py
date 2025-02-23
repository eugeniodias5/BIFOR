import os, json
from PIL import Image

from torch.utils.data import Dataset


class SibetanDataset(Dataset):
    def __init__(self, root, gt_path, transform=None):
        self.root = root
        self.transform = transform

        self.targets = []
        self.samples = []

        # Importing the ground truth labels
        with open(gt_path) as f:
            clus_ground_truth = json.load(f)

        dog_labels = []
        gt_dict = {}
        self.clus_labels = []

        for key in clus_ground_truth.keys():
            for lab in clus_ground_truth[key]:
                dog_labels.append(lab)
                gt_dict[lab] = int(key)

        self.folder_labels = sorted([int(label) for label in os.listdir(root) if os.path.isdir(os.path.join(root, label))])
        
        for label in self.folder_labels:
            if dog_labels is not None:
                if label not in dog_labels:
                    continue

            for img in os.listdir(os.path.join(root, str(label))):
                self.targets.append(label)
                self.samples.append(os.path.join(root, str(label), img))
            
            self.clus_labels.append(gt_dict[label])
    
    def num_classes(self):
        return len(set(self.targets))

    def __len__(self):
        return len(self.targets)
    
    def __getitem__(self, idx):
        # Return image and label
        img = Image.open(self.samples[idx])
        label = self.targets[idx]

        if self.transform:
            img = self.transform(img)
        
        return img, label