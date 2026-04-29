import os
import numpy as np
from torch.utils.data import Dataset

class MP4Dataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.samples = []
        self.labels = []
        self._load_data()

    def _load_data(self):
        for label, folder in enumerate(['0', '1']):
            folder_path = os.path.join(self.root_dir, folder)
            files = os.listdir(folder_path)
            for file in files:
                if file.endswith('.npy'):
                    file_path = os.path.join(folder_path, file)
                    self.samples.append(file_path)
                    self.labels.append(label)
        self.samples = np.array(self.samples)
        self.labels = np.array(self.labels)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_path = self.samples[idx]
        label = self.labels[idx]
        sample = np.load(sample_path)
        return sample, label