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

#from misc import utils
#from misc import metric
#from misc import options
#from dataset.dataset_kitti2017 import DatasetKITTI2017
#from dataset.dataset_kitti2015 import DatasetKITTI2015
from dataset import vaihingen_evaluation_ex as ve
from dataset import vaihingen_info as info

from model.psmnet_lidar import PSMNetLiDAR
from model.psmnet_lidar_deux import PSMNetLiDARDeux
from model.psmnet_lidar_guide import PSMNetLiDARGuide
from model.psmnet_lidar_guide_full import PSMNetLiDARGuideFull
from model.psmnet_lidar_2x import PSMNetLiDAR2x
from model.psmnet_lidar_deux_2x import PSMNetLiDARDeux2x

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
    parser.add_argument('--datalist', default='', help='input data image list')
    parser.add_argument('--info_datapath', default='', help='information path')
    parser.add_argument('--guideL', default='guide0_05', help='information path')
    parser.add_argument('--guideR', default='guideR0_05', help='information path')
    parser.add_argument('--savepath', default='', help='save result directory')
    parser.add_argument('--subfolder', default='', help='save result with subfolder')
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

    print('***[args]: ', args)

    print('torch version: ', torch.__version__)
    print('cuda version: ', torch.version.cuda)
    #exit()

    #model = options.get_model(args.model_name)
    normalize = info.load_vaihingen_info(args.info_datapath)

    #PSMNet_LIDAR
    norm_mode = ['naive_categorical', # Applying categorical CBN on 3D-CNN in stereo matching network
                'naive_continuous', # Applying continuous CBN on 3D-CNN in stereo matching network
                'categorical', # Applying categorical CCVNorm on 3D-CNN in stereo matching network
                'continuous', # Applying continuous CCVNorm on 3D-CNN in stereo matching network
                'categorical_hier', # Applying categorical HierCCVNorm on 3D-CNN in stereo matching network
                ][4]

    if args.model_name == 'basic':
        print('load basic model')
        model = PSMNetLiDAR(maxdisparity = args.maxdisp, norm_mode = norm_mode)
    elif args.model_name == 'deux':
        print('load deux model')
        model = PSMNetLiDARDeux(maxdisparity = args.maxdisp, norm_mode = norm_mode)
    elif args.model_name == 'guide':
        print('load guide model')
        model = PSMNetLiDARGuide(maxdisparity = args.maxdisp, norm_mode = norm_mode)
    elif args.model_name == 'full':
        print('load full model')
        model = PSMNetLiDARGuideFull(maxdisparity = args.maxdisp, norm_mode = norm_mode)
    elif args.model_name == 'basic2x':
        print('load basic2x model')
        model = PSMNetLiDAR2x(maxdisparity = args.maxdisp, norm_mode = norm_mode)
    elif args.model_name == 'deux2x':
        print('load deux2x model')
        model = PSMNetLiDARDeux2x(maxdisparity = args.maxdisp, norm_mode = norm_mode)
    else:
        print('no model')

    if not args.no_cuda:
        # make no sense
        model = nn.DataParallel(model)
        model = model.cuda()
    
    # utils.load_checkpoint(model, None, None, args.model_path, True)
    # load model
    if args.model_path is not None:
        print('load PSMNet LiDAR')
        state_dict = torch.load(args.model_path)
        model.load_state_dict(state_dict['state_dict'])

    # Define testing dataset (NOTE: currently using validation set)
    all_left_img, all_right_img, all_left_guide, all_right_guide, all_left_disp = ve.load_vaihingen_evluation(args.datalist, args.guideL, args.guideR, args.savepath, args.subfolder)
    
    # Perform testing
    model.eval()
    #pbar = tqdm(all_left_img)
    #pbar.set_description('Testing')
    
    infer_time = 0
    # valuation transform
    transform_rgb = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(**normalize)
    ])
    transform_depth = transforms.Compose([
        transforms.ToPILImage(mode='F'), # NOTE: is this correct?!
        transforms.ToTensor()
    ])

    with torch.no_grad():
        #for it, left in enumerate(pbar):
        for it, left in enumerate(all_left_img):
            imgL_o = Image.open(all_left_img[it]).convert('RGB')
            imgR_o = Image.open(all_right_img[it]).convert('RGB')
            #imgL = io.imread(all_left_img[it])
            #imgR = io.imread(all_right_img[it])
            guideL_o = Image.open(all_left_guide[it])
            guideR_o = Image.open(all_right_guide[it])

            # test crop
            # large image, there is an error
            """imgL_o = imgL_o.crop((0, 0, 768, 1024))
            imgR_o = imgR_o.crop((0, 0, 768, 1024))
            guideL_o = guideL_o.crop((0, 0, 768, 1024))
            guideR_o = guideR_o.crop((0, 0, 768, 1024))"""

            imgL = np.array(imgL_o)
            imgR = np.array(imgR_o)
            guideL = np.array(guideL_o).astype(np.float32)/args.disp_scale#np.ascontiguousarray(guideL_o,dtype=np.float32)/args.disp_scale
            guideL = guideL[:,:,np.newaxis]
            guideR = np.array(guideR_o).astype(np.float32)/args.disp_scale#np.ascontiguousarray(guideR_o,dtype=np.float32)/args.disp_scale
            guideR = guideR[:,:,np.newaxis]
            
            # Pack data
            inputs = dict()
            inputs['left_rgb'] = torch.unsqueeze(transform_rgb(imgL),0) 
            inputs['right_rgb'] = torch.unsqueeze(transform_rgb(imgR),0) 
            inputs['left_sd'] = torch.unsqueeze(transform_depth(guideL),0) 
            inputs['right_sd'] = torch.unsqueeze(transform_depth(guideR),0) 
            
            if not args.no_cuda :
                inputs['left_rgb'] = inputs['left_rgb'].cuda()
                inputs['right_rgb'] = inputs['right_rgb'].cuda()
                inputs['left_sd'] = inputs['left_sd'].cuda()
                inputs['right_sd'] = inputs['right_sd'].cuda()

            #print('image: ')
            #print(inputs['left_rgb'].shape)
            #print('depth: ')
            #print(inputs['left_sd'].shape)

            # Inference
            end = time.time()
            pred = model(inputs)

            pred_disp_img = (torch.squeeze(pred)).data.cpu().detach().numpy()
            pred_disp_img = (pred_disp_img * 256).astype('uint16')
            #print(pred_disp_img)
            save_path, filename = os.path.split(all_left_disp[it])
            if not os.path.exists(save_path) :
                os.makedirs(save_path)
            
            #print('save: ' + all_left_disp[i])
            imageio.imwrite(all_left_disp[it], pred_disp_img)
            
            infer_time += (time.time() - end)

            # refer https://discuss.pytorch.org/t/how-can-i-release-the-unused-gpu-memory/81919
            # result is bad
            #torch.cuda.empty_cache()
    infer_time /= len(all_left_img)

    print('Average infer time: {}'.format(infer_time))

if __name__ == '__main__':
    main()
