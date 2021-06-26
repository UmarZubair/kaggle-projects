import glob
import pandas as pd
import numpy as np
import torch
from PIL import Image


class EthiopicDataset:
    def __init__(self, df, transforms=None):
        self.image_paths = df['image_path']
        self.target = df['target']
        self.transforms = transforms

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx])
        image = np.array(image).astype('float32')
        target = self.target[idx]
        if self.transforms is not None:
            image = self.transforms(image)
        return image, torch.tensor(target).long()


class TestEthioipic:
    def __init__(self, image_path, transforms):
        self.image_path = image_path
        self.transforms = transforms

    def __len__(self):
        return len(self.image_path)

    def __getitem__(self, idx):
        image = Image.open(self.image_path[idx])
        image = np.array(image).astype('float32')
        image = self.transforms(image)
        return image


def make_csv(data_path):
    image_path = sorted(glob.glob(data_path + '/*/*.jpg'))
    targets = []
    target = [1 - 1, 10 - 1, 2 - 1, 3 - 1, 4 - 1, 5 - 1, 6 - 1, 7 - 1, 8 - 1, 9 - 1]
    [targets.extend(6000 * [i]) for i in target]
    df = pd.DataFrame(list(zip(image_path, targets)), columns=['image_path', 'target'])
    df.to_csv('check.csv')
    return df
