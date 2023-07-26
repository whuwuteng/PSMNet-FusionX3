import os
import torch
import torch.utils.data as data
import torch
import torchvision.transforms as transforms
import random
from PIL import Image, ImageOps
import numpy as np
from dataset  import preprocess 

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

# convert to RGB
def default_loader(path):
    return Image.open(path).convert('RGB')

def gray_loader(path):
    return Image.open(path).convert('L')
    
def disparity_loader(path):
    return Image.open(path)

# change or not, not sure
# use an adapter to lightning
class TestDatasetToulouseAdapter(data.Dataset):
    def __init__(self, left, right, left_guide, right_guide, left_disparity, loader=default_loader, dploader= disparity_loader, normalize=None, disp_scale=256):
 
        self.left = left
        self.right = right
        self.left_guide = left_guide
        self.right_guide = right_guide

        self.disp_L = left_disparity

        self.loader = loader
        self.dploader = dploader

        self.normalize = normalize
        self.disp_scale = disp_scale

    def __getitem__(self, index):
        left  = self.left[index]
        right = self.right[index]

        left_guide = self.left_guide[index]
        right_guide = self.right_guide[index]

        #print(left)

        left_rgb = self.loader(left)
        left_rgb = np.array(left_rgb)
        right_rgb = self.loader(right)
        right_rgb = np.array(right_rgb)

        #w, h = left_rgb.size

        left_sdepth = self.dploader(left_guide)
        left_sdepth = np.array(left_sdepth).astype(np.float32) / self.disp_scale

        right_sdepth = self.dploader(right_guide)
        right_sdepth = np.array(right_sdepth).astype(np.float32) / self.disp_scale

        # need pad or not
        transform_rgb = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(**self.normalize)
        ])
        transform_depth = transforms.Compose([
        transforms.ToPILImage(mode='F'), # NOTE: is this correct?!
        transforms.ToTensor()
        ])

        data = dict()
        
        data['left_rgb'], data['right_rgb'] = list(map(transform_rgb, [left_rgb, right_rgb]))
        data['left_sd'], data['right_sd'] = list(map(transform_depth, [left_sdepth, right_sdepth]))

        # want to save the name
        data['disp'] = self.disp_L[index]
        
        return data

    def __len__(self):
        return len(self.left)
