import torch
from torch.utils.data import Dataset
import os
from PIL import Image
from torchvision import transforms
import numpy as np
from datasets.sequence_aug import *


class dataset(Dataset):

    def __init__(self, list_data, test=False, transform=None):
        self.test = test
        if self.test:
            self.seq_data = list_data['data'].tolist()
        else:
            self.seq_data = list_data['data'].tolist()
            self.labels = list_data['label'].tolist()
            self.seq_data_tensor = self.seq_data
        if transform is None:
            self.transforms = Compose([
                Reshape()
            ])
        else:
            self.transforms = transform
        self.R = None

    def __len__(self):
        return len(self.seq_data)

    def __getitem__(self, item):
        if self.test:
            seq = self.seq_data[item]
            seq = self.transforms(seq)
            return seq, item
        else:
            seq = self.seq_data[item]
            label = self.labels[item]
            seq = self.transforms(seq)
            return seq, label


