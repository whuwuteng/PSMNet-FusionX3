from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F
import numpy as np
import time
import math

#from model.psmnet_ex import PSMNetEx
from model.stackhourglass_2x import PSMNet2x
from model.psmnet_lidar_guide_full import PSMNetLiDARGuideFull

#import cv2
# a bug in PIL: cannot write mode I;16 as PNG
from PIL import Image

import pdb
import imageio
from dataset import vaihingen_evaluation as ve
from dataset import vaihingen_info as info

# 2012 data /media/jiaren/ImageNet/data_scene_flow_2012/testing/

parser = argparse.ArgumentParser(description='PSMNet')
parser.add_argument('--datalist', default='', help='input data image list')
parser.add_argument('--info_datapath', default=None,
                    help='information path')                    
parser.add_argument('--loadmodel', default='./trained/pretrained_model_KITTI2015.tar',
                    help='loading model')
parser.add_argument('--savepath', default='', help='save result directory')
parser.add_argument('--subfolder', default='', help='save result with subfolder')
parser.add_argument('--model', default='stackhourglass',
                    help='select model')
parser.add_argument('--maxdisp', type=int, default=192,
                    help='maxium disparity')
parser.add_argument('--disp_scale', type=int ,default=256,
                    help='maxium disparity') 
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

print('***[args]: ', args)

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

model = PSMNet2x(args.maxdisp)
#model = PSMNetLiDARGuideFull(args.maxdisp)

model = nn.DataParallel(model, device_ids=[0])
model.cuda()

#for name, layer in model.named_modules():
#    print(name, layer)
#exit()

all_left_img, all_right_img, all_left_disp = ve.load_vaihingen_evluation(args.datalist, args.savepath, args.subfolder)

#pdb.set_trace()

normal_mean_var = info.load_vaihingen_info(args.info_datapath)

if args.loadmodel is not None:
    print('load PSMNet')
    state_dict = torch.load(args.loadmodel)
    model.load_state_dict(state_dict['state_dict'])

print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

def test(imgL,imgR):
    model.eval()

    if args.cuda:
        imgL = imgL.cuda()
        imgR = imgR.cuda()     

    #pdb.set_trace()

    with torch.no_grad():
        disp = model(imgL,imgR)

    disp = torch.squeeze(disp)
    pred_disp = disp.data.cpu().numpy()

    return pred_disp


def main():

    print('[***] norm= ', normal_mean_var)
    infer_transform = transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize(**normal_mean_var)])    
    num = len(all_left_img)
    
    for i in range(0, num):
        if os.path.isfile(all_left_img[i]) and os.path.isfile(all_right_img[i]) :
            imgL_o = Image.open(all_left_img[i]).convert('RGB')
            imgR_o = Image.open(all_right_img[i]).convert('RGB')

            imgL = infer_transform(imgL_o)
            imgR = infer_transform(imgR_o) 
            
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

            start_time = time.time()
            pred_disp = test(imgL,imgR)
            print('time = %.2f' %(time.time() - start_time))

            if top_pad !=0 and right_pad != 0:
                img = pred_disp[top_pad:,:-right_pad]
            else:
                img = pred_disp

            img = (img*args.disp_scale).astype('uint16')

            save_path, filename = os.path.split(all_left_disp[i])
            if not os.path.exists(save_path) :
                os.makedirs(save_path)
                
            #print('save: ' + all_left_disp[i])
            imageio.imwrite(all_left_disp[i], img)

if __name__ == '__main__':
   main()