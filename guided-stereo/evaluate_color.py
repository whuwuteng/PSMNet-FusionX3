from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import time
import math
from models import *
#import cv2
# a bug in PIL: cannot write mode I;16 as PNG
from PIL import Image

import pdb
import imageio
from dataloader import vaihingen_evaluation as ve
from dataloader import preprocess
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
parser = argparse.ArgumentParser(description='Guided-Stereo')
parser.add_argument('--datalist', default='', help='input data image list')
parser.add_argument('--loadmodel', default='./trained/pretrained_model_KITTI2015.tar',
                    help='loading model')
parser.add_argument('--savepath', default='', help='save result directory')
parser.add_argument('--subfolder', default='', help='save result with subfolder')
parser.add_argument('--maxdisp', type=int, default=192,
                    help='maxium disparity')
parser.add_argument('--disp_scale', type=int ,default=256,
                    help='maxium disparity') 
parser.add_argument('--guidefolder', default='guide', help='guided disparity folder')
parser.add_argument('--guided', action='store_true', default=False, help='Enable guided stereo')
parser.add_argument('--display', action='store_true', default=False, help='Display output')
parser.add_argument('--save', action='store_true', default=False, help='Save output')
parser.add_argument('--verbose', action='store_true', default=False, help='Print stats for each single image')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')

args = parser.parse_args()
print('[***] args= ', args) 

args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

model = psmnet(args.maxdisp, args.guided)

if args.cuda :
    #model = nn.DataParallel(model, device_ids=[0])
    model = nn.DataParallel(model)
    model.cuda()

if args.loadmodel is not None:
    print('load Guide-Stereo')
    state_dict = torch.load(args.loadmodel)
    model.load_state_dict(state_dict['state_dict'])
else :
    print('no model')
    exit()

print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

all_left_img, all_right_img, all_left_guide, all_left_disp = ve.load_vaihingen_evluation(args.datalist, args.savepath, args.subfolder, args.guidefolder)

#pdb.set_trace()

# Running guided stereo!
def test(imgL,imgR,guideL,h,w):
    model.eval()
    #imgL   = Variable(torch.FloatTensor(imgL))
    #imgR   = Variable(torch.FloatTensor(imgR))   
    guideL = Variable(torch.FloatTensor(guideL)) 
    guideL = guideL.unsqueeze(0)

    #print(validhints.size())
    validhints = (guideL > 0).float()

    #computing density
    density=(float(torch.nonzero(validhints).size(0)) / ((validhints.size(1))*validhints.size(2))*100.)
    print(density)

    if args.cuda:
        imgL, imgR = imgL.cuda(), imgR.cuda()

    with torch.no_grad():
        output3 = model(imgL,imgR,guideL,validhints,k=10,c=1)
        pred_disp = torch.squeeze(output3)
        pred_disp = pred_disp.data.cpu().numpy()

    return pred_disp

def main():
    # set transform
    processed = preprocess.get_transform(augment=False)
    num = len(all_left_img)
    
    for i in range(0, num):
        if os.path.isfile(all_left_img[i]) and os.path.isfile(all_right_img[i]) :
            imgL_o = Image.open(all_left_img[i]).convert('RGB')
            imgR_o = Image.open(all_right_img[i]).convert('RGB')
            guideL_o = Image.open(all_left_guide[i])

            guideL = np.ascontiguousarray(guideL_o,dtype=np.float32)/args.disp_scale
            
            w, h = imgL_o.size

            imgL = processed(imgL_o)
            imgR = processed(imgR_o)

            if imgL.shape[1] % 16 != 0:
                times = imgL.shape[1]//16       
                top_pad = (times+1)*16 -imgL.shape[1]
            else:
                top_pad = 0

            if imgL.shape[2] % 16 != 0:
                times = imgL.shape[2]//16                       
                right_pad = (times+1)*16-imgL.shape[2]
            else:
                right_pad = 0    

            imgL = F.pad(imgL,(0,right_pad, top_pad,0)).unsqueeze(0)
            imgR = F.pad(imgR,(0,right_pad, top_pad,0)).unsqueeze(0)
            guideL = np.pad(guideL, ((0, right_pad), (top_pad, 0)), mode='constant', constant_values=0)
            #print(guideL)
            #guideL = F.pad(guideL,(0,right_pad, top_pad,0)).unsqueeze(0)

            start_time = time.time()
            pred_disp = test(imgL,imgR,guideL,h,w)
            
            print('time = %.2f' %(time.time() - start_time))

            if top_pad !=0 and right_pad != 0:
                img = pred_disp[top_pad:,:-right_pad]
            else:
                img = pred_disp

            img = (img*args.disp_scale).astype('uint16')

            save_path, filename = os.path.split(all_left_disp[i])
            if not os.path.exists(save_path) :
                os.makedirs(save_path)
            
            #print(img)
            #print('save: ' + all_left_disp[i])
            imageio.imwrite(all_left_disp[i], img)

if __name__ == '__main__':
   main()
