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
class TestImageFloderStdAdapter(data.Dataset):
    def __init__(self, left, right, left_disparity, loader=default_loader, normalize=None):
 
        self.left = left
        self.right = right
        self.disp_L = left_disparity
        self.loader = loader

        self.normalize = normalize

    def __getitem__(self, index):
        data = {}

        left  = self.left[index]
        right = self.right[index]

        #print(left)

        left_img = self.loader(left)
        right_img = self.loader(right)

        w, h = left_img.size

        # need pad or not
        processed = preprocess.get_transform(normalize=self.normalize, augment=False, channle=len(left_img.getbands()))  
        left_img       = processed(left_img)
        right_img      = processed(right_img)

        #return left_img, right_img, dataL
        data['left'] = left_img
        data['right'] = right_img

        # want to save the name
        data['disp'] = self.disp_L[index]
        
        return data

    def __len__(self):
        return len(self.left)
