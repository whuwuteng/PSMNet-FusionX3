import os
import torch
import torch.utils.data as data
import torch
import torchvision.transforms as transforms
import random
from PIL import Image, ImageOps
import numpy as np
from dataloader  import preprocess 

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


class imageLoader(data.Dataset):
    def __init__(self, left, right, guide, left_disparity, training, loader=default_loader, dploader= disparity_loader, disp_scale = 256):
 
        self.left = left
        self.right = right
        self.guide = guide
        self.disp_L = left_disparity
        self.loader = loader
        self.dploader = dploader
        self.training = training
        self.disp_scale = disp_scale

    def __getitem__(self, index):
        left  = self.left[index]
        right = self.right[index]
        disp_L= self.disp_L[index]
        guide = self.guide[index]

        left_img = self.loader(left)
        right_img = self.loader(right)
        dataL = self.dploader(disp_L)
        guide_img = self.dploader(guide)

        if self.training:  
           w, h = left_img.size
           th, tw = 256, 512
 
           x1 = random.randint(0, w - tw)
           y1 = random.randint(0, h - th)

           left_img = left_img.crop((x1, y1, x1 + tw, y1 + th))
           right_img = right_img.crop((x1, y1, x1 + tw, y1 + th))

           dataL = np.ascontiguousarray(dataL,dtype=np.float32)/self.disp_scale
           dataL = dataL[y1:y1 + th, x1:x1 + tw]
           
           guideL = np.ascontiguousarray(guide_img,dtype=np.float32)/self.disp_scale
           guideL = guideL[y1:y1 + th, x1:x1 + tw]
           processed = preprocess.get_transform(augment=False)
           
           rawimage = preprocess.identity(256)
           
           reference  = rawimage(left_img)
           left_img   = processed(left_img)
           right_img  = processed(right_img)

           return reference, left_img, right_img, guideL, dataL, h, w
        else:
           w, h = left_img.size
           
           left_img = left_img.crop((w-1280, h-384, w, h))
           right_img = right_img.crop((w-1280, h-384, w, h))
           w1, h1 = left_img.size

           dataL = dataL.crop((w-1280, h-384, w, h))
           dataL = np.ascontiguousarray(dataL,dtype=np.float32)/256

           guideL = guide_img.crop((w-1280, h-384, w, h))
           guideL = np.ascontiguousarray(guideL,dtype=np.float32)/256
           
           processed = preprocess.get_transform(augment=False) 
           rawimage  = preprocess.identity(256)
           
           reference      = rawimage(left_img)
           left_img       = processed(left_img)
           right_img      = processed(right_img)

           return reference, left_img, right_img, guideL, dataL, h, w

    def __len__(self):
        return len(self.left)
