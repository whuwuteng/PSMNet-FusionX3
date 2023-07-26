import os
import torch
from torch.utils.data import DataLoader
import torch
import torchvision.transforms as transforms
import random
from PIL import Image, ImageOps
import numpy as np
from dataset import preprocess 
from dataset.TestLoader_std_adapter import TestImageFloderStdAdapter

import pytorch_lightning as pl

# there is an adapter class
class TestImageFloderStd_lightning(pl.LightningDataModule):
    def __init__(self, left_test, right_test, disp_test, normalize, test_batch_size = 1, num_workers = 8):
        super().__init__()

        self.left_test = left_test
        self.right_test = right_test
        self.disp_test =disp_test
        
        self.normalize = normalize

        self.test_batch_size = test_batch_size

        self.num_workers = num_workers

    # load the dataset
    def setup(self, stage=None):
        #print('********: ', stage)
        if stage == 'test' or stage is None:
            self.testset = TestImageFloderStdAdapter(self.left_test, self.right_test, self.disp_test, normalize=self.normalize)
        else:
            raise NotImplementedError

    def test_dataloader(self):
        return DataLoader(self.testset, batch_size=self.test_batch_size, num_workers=self.num_workers, shuffle=False)


