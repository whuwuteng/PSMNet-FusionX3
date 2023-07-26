import os
import torch
from torch.utils.data import DataLoader
import torch
import torchvision.transforms as transforms
import random
from PIL import Image, ImageOps
import numpy as np
from dataset import preprocess 
from dataset.dataset_toulouseEx_weight import DatasetToulouseWeight

import pytorch_lightning as pl

# there is an adapter class
class DatasetToulouseWeight_lightning(pl.LightningDataModule):
    def __init__(self, datalist_train, train_output_size, datalist_val, val_output_size, guideL_folder, guideR_folder, normalize, train_batch_size = 12, val_batch_size = 4, num_workers = 4):
        super().__init__()
        self.datalist_train = datalist_train
        self.train_output_size = train_output_size

        self.datalist_val = datalist_val
        self.val_output_size = val_output_size

        self.guideL_folder = guideL_folder
        self.guideR_folder = guideR_folder
        
        self.normalize = normalize

        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.num_workers = num_workers

    # load the dataset
    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.trainset = DatasetToulouseWeight(self.datalist_train, self.train_output_size, 'train', self.guideL_folder, self.guideR_folder, normalize=self.normalize)
            self.valset = DatasetToulouseWeight(self.datalist_val, self.val_output_size, 'val', self.guideL_folder, self.guideR_folder, normalize=self.normalize)
        else:
            raise NotImplementedError

    def train_dataloader(self):
        return DataLoader(self.trainset, batch_size=self.train_batch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.valset, batch_size=self.val_batch_size, num_workers=self.num_workers, shuffle=False)
