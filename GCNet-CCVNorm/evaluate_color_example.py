"""
Testing process.

Usage:
# For KITTI Depth Completion
>> python test.py --model_cfg exp/test/test_options.py --model_path exp/test/ckpt/\[ep-00\]giter-0.ckpt \
                  --dataset kitti2017 --rgb_dir ./data/kitti2017/rgb --depth_dir ./data/kitti2015/depth
# For KITTI Stereo
>> python test.py --model_cfg exp/test/test_options.py --model_path exp/test/ckpt/\[ep-00\]giter-0.ckpt \
                  --dataset kitti2015 --root_dir ./data/kitti_stereo/data_scene_flow
"""

import os
import sys
import time
import argparse
import importlib
import random
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.backends.cudnn as cudnn
from PIL import Image
import imageio

from misc import utils
from misc import metric
from dataset.dataset_kitti2017 import DatasetKITTI2017
from dataset.dataset_kitti2015 import DatasetKITTI2015
from dataset import vaihingen_evaluation as ve
from misc import options

from model.psmnet_lidar import PSMNetLiDAR

SEED = 100
random.seed(SEED)
np.random.seed(seed=SEED)
cudnn.deterministic = True
cudnn.benchmark = False
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

def parse_arg():
    parser = argparse.ArgumentParser(description='Sparse-Depth-Stereo testing')
    parser.add_argument('--model_name', dest='model_name', type=str, default='',
                        help='Configuration model nameof the trained model.')
    parser.add_argument('--model_path', dest='model_path', type=str, default=None,
                        help='Path to weight of the trained model.')
    parser.add_argument('--leftimg', default='', help='input left image')
    parser.add_argument('--rightimg', default='', help='input right image')
    parser.add_argument('--leftGuide', default=None, help='input left guide image')
    parser.add_argument('--rightGuide', default=None, help='input right guide image')
    parser.add_argument('--result', default='', help='save result with subfolder')
    parser.add_argument('--maxdisp', type=int, default=192,
                        help='maxium disparity')
    parser.add_argument('--disp_scale', type=int ,default=256,
                        help='maxium disparity') 
    parser.add_argument('--no_cuda', dest='no_cuda', action='store_true',
                            help='Don\'t use gpu')
    parser.set_defaults(no_cuda=False)
    args = parser.parse_args()
    return args

# test can be run for the memeory problem
# can not train on pytorch 1.7.1 and test on 1.2.0
# test on pytorch 1.2.0, there is a bug
# File "/home/qt/tengw/pycode/Stereo-LiDAR-CCVNorm/model/gcnet_lidar.py", line 167, in forward

# tmp = Variable(torch.arange(0, out.shape[1]).type_as(out.data))
# RuntimeError: CUDA error: an illegal memory access was encountered

# lack memory is still a problem
def main():
    # Parse arguments
    args = parse_arg()

    # Define model and load
    model = options.get_model(args.model_name)

    if not args.no_cuda:
        # make no sense
        # not work for the GC_net_LIDAR
        #model = nn.DataParallel(model)
        model = model.cuda()
    utils.load_checkpoint(model, None, None, args.model_path, True)

    # Perform testing
    model.eval()
    #pbar = tqdm(all_left_img)
    #pbar.set_description('Testing')
    
    infer_time = 0
    # valuation transform
    transform_rgb = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor()
    ])
    transform_depth = transforms.Compose([
        transforms.ToPILImage(mode='F'), # NOTE: is this correct?!
        transforms.ToTensor()
    ])

    with torch.no_grad():
        #for it, left in enumerate(pbar):
        imgL_o = Image.open(args.leftimg).convert('RGB')
        imgR_o = Image.open(args.rightimg).convert('RGB')

        # test for 16GB GPU 
        # refer to https://stackoverflow.com/questions/71738218/module-pil-has-not-attribute-resampling
        imgL_o.thumbnail((512,512), Image.BICUBIC)
        imgR_o.thumbnail((512,512), Image.BICUBIC)

        imgL = np.array(imgL_o)
        imgR = np.array(imgR_o)
        if args.leftGuide and args.rightGuide :
            guideL_o = Image.open(args.leftGuide)
            guideR_o = Image.open(args.rightGuide)

            guideL_o.thumbnail((512,512), Image.NEAREST)
            guideR_o.thumbnail((512,512), Image.NEAREST)

            guideL = np.array(guideL_o).astype(np.float32)/args.disp_scale/2.0#np.ascontiguousarray(guideL_o,dtype=np.float32)/args.disp_scale
            guideL = guideL[:,:,np.newaxis]
            guideR = np.array(guideR_o).astype(np.float32)/args.disp_scale/2.0#np.ascontiguousarray(guideR_o,dtype=np.float32)/args.disp_scale
            guideR = guideR[:,:,np.newaxis]
        
        # Pack data
        inputs = dict()
        inputs['left_rgb'] = torch.unsqueeze(transform_rgb(imgL),0) 
        inputs['right_rgb'] = torch.unsqueeze(transform_rgb(imgR),0) 

        if args.leftGuide and args.rightGuide :
            inputs['left_sd'] = torch.unsqueeze(transform_depth(guideL),0) 
            inputs['right_sd'] = torch.unsqueeze(transform_depth(guideR),0) 
        
        if not args.no_cuda :
            inputs['left_rgb'] = inputs['left_rgb'].cuda()
            inputs['right_rgb'] = inputs['right_rgb'].cuda()
            if args.leftGuide and args.rightGuide :
                inputs['left_sd'] = inputs['left_sd'].cuda()
                inputs['right_sd'] = inputs['right_sd'].cuda()

        # Inference
        end = time.time()
        pred = model(inputs)

        pred_disp_img = (torch.squeeze(pred)).data.cpu().detach().numpy()
        pred_disp_img = (pred_disp_img * 256).astype('uint16')
        #print(pred_disp_img)
        save_path, filename = os.path.split(args.result)
        if not os.path.exists(save_path) :
            os.makedirs(save_path)
        
        #print('save: ' + all_left_disp[i])
        imageio.imwrite(args.result, pred_disp_img)
        
        infer_time += (time.time() - end)

    print('infer time: {}'.format(infer_time))

if __name__ == '__main__':
    main()
