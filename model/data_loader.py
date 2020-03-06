#!/usr/bin/env python3
"""Define custom dataset class extending the Pytorch Dataset class
"""

import pandas as pd
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as tx
import torchvision.datasets as ds

# Add transformations into the constants below.
TRAIN_TRANSFORMER = tx.Compose([tx.Resize(28),
                                tx.RandomHorizontalFlip(),
                                tx.ToTensor(),
                                tx.Normalize((0.1307,), (0.3081,))])
EVAL_TRANSFORMER = tx.Compose([tx.Resize(28),
                               tx.ToTensor(),
                               tx.Normalize((0.1307,), (0.3081,))])


class MyDataset(Dataset):
    """Define a dataset in PyTorch.
    """
    def __init__(self, data_path, transform=None):
        """Get the filenames and labels of images from a csv file.
        Args:
            data_path: (string) directory containing the dataset
            transform: (torchvision.transforms) transformation to apply on images
        """
        self.data = pd.read_csv(data_path)
        self.transform = transform

    def __len__(self):
        """Return the size of the dataset.
        """
        return len(self.data)

    def __getitem__(self, idx):
        """Get an image and label at index idx from the dataset. Perform transforms on image.
        Args:
            idx: (int) index in [0, 1, ..., size_of_dataset-1]
        Returns:
            image: (torch.Tensor) transformed image
            label: (int) corresponding label of image
        """
        image = Image.open(self.data[idx, 0])
        if self.transform:
            image = self.transform(image)
        label = self.data[idx, 1]
        return image, label


def get_dataloader(modes, data_dir, params):
    """Get DataLoader objects from data_dir.
    Args:
        modes: (list) mode of operation i.e. 'train', 'val', 'test'
        data_dir: (string) directory containing the dataset
        params: (Params) hyperparameters
    Returns:
        dataloaders: (dict) DataLoader object for each mode
    """
    dataloaders = {}

    for mode in modes:
        if mode == 'train':
            d_l = DataLoader(ds.MNIST(root=data_dir,
                                      train=True,
                                      download=False,
                                      transform=TRAIN_TRANSFORMER),
                             batch_size=params.batch_size,
                             shuffle=True,
                             num_workers=params.num_workers)
        else:
            d_l = DataLoader(ds.MNIST(root=data_dir,
                                      train=False,
                                      download=False,
                                      transform=EVAL_TRANSFORMER),
                             batch_size=params.batch_size,
                             shuffle=False,
                             num_workers=params.num_workers)

        dataloaders[mode] = d_l

    return dataloaders
