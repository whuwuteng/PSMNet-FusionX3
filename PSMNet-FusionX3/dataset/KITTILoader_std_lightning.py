import os
import torch
from torch.utils.data import DataLoader
import torch
import torchvision.transforms as transforms
import random
from PIL import Image, ImageOps
import numpy as np
from dataset import preprocess 
from dataset.KITTILoader_std_adapter import myImageFloderStdAdapter

import pytorch_lightning as pl

# there is an adapter class
class myImageFloderStd_lightning(pl.LightningDataModule):
    def __init__(self, left_train, right_train, disp_train, left_val, right_val, disp_val, normalize, train_batch_size = 12, val_batch_size = 4, num_workers = 8):
        super().__init__()
        self.left_train = left_train
        self.right_train = right_train
        self.disp_train = disp_train

        self.left_val = left_val
        self.right_val = right_val
        self.disp_val = disp_val
        
        self.normalize = normalize

        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.num_workers = num_workers

    # load the dataset
    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.trainset = myImageFloderStdAdapter(self.left_train, self.right_train, self.disp_train, True, normalize=self.normalize)
            self.valset = myImageFloderStdAdapter(self.left_val, self.right_val, self.disp_val, False, normalize=self.normalize)
        else:
            raise NotImplementedError

    def train_dataloader(self):
        return DataLoader(self.trainset, batch_size=self.train_batch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.valset, batch_size=self.val_batch_size, num_workers=self.num_workers, shuffle=False)
