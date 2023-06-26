from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
#from torch.nn.parallel import DistributedDataParallel as DDP
#import torch.distributed as dist

#import skimage
#import skimage.io
#import skimage.transform
import numpy as np
import time
import math
from dataset import vaihingen_collector_file as vse
from dataset import KITTILoader_std as DA
from dataset import vaihingen_info as info

from model.psmnet_lidar import PSMNetLiDAR
from model.psmnet_lidar_deux import PSMNetLiDARDeux
from model.psmnet_lidar_guide import PSMNetLiDARGuide
from model.psmnet_lidar_guide_full import PSMNetLiDARGuideFull
from model.psmnet_lidar_2x import PSMNetLiDAR2x
from model.psmnet_lidar_deux_2x import PSMNetLiDARDeux2x

from dataset.dataset_toulouseEx import DatasetToulouse

import pdb

parser = argparse.ArgumentParser(description='PSMNet')
parser.add_argument('--maxdisp', type=int ,default=192,
                    help='maxium disparity')
parser.add_argument('--model', default='basic',
                    help='select model')
parser.add_argument('--datatype', default='2015',
                    help='datapath')
parser.add_argument('--datapath', default='/media/jiaren/ImageNet/data_scene_flow_2015/training/',
                    help='datapath')
parser.add_argument('--train_datapath', default=None,
                    help='training data path')
parser.add_argument('--val_datapath', default=None,
                    help='validation data path')
parser.add_argument('--info_datapath', default='',
                    help='information path')
parser.add_argument('--guideL', default='guide0_05',
                    help='information path')
parser.add_argument('--guideR', default='guideR0_05',
                    help='information path')
parser.add_argument('--disp_scale', type=int , default=256,
                    help='maxium disparity')
parser.add_argument('--epoch_start', type=int, default=0,
                    help='number of epochs to train for')
parser.add_argument('--epochs', type=int, default=300,
                    help='number of epochs to train')
parser.add_argument('--loadmodel', default=None,
                    help='load model')
parser.add_argument('--resume', action='store_true',
                    default=False, help='load model and optimizer')
parser.add_argument('--savemodel', default='./',
                    help='save model')
parser.add_argument('--no-cuda', action='store_true',
                    default=False, help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')

args = parser.parse_args()

print('***[args]: ', args)

# not really work
# too complicate
#torch.distributed.init_process_group(backend="nccl", rank=0, world_size=2)

args.cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)

if args.cuda:
    torch.cuda.manual_seed(args.seed)

normalize = info.load_vaihingen_info(args.info_datapath)

train_output_size = (256, 512)
val_output_size =  (256, 512) # NOTE: set to (256, 1216) if there is enough gpu memory
train_dataset = DatasetToulouse(args.train_datapath, train_output_size, 'train', args.guideL, args.guideR, normalize)
val_dataset = DatasetToulouse(args.val_datapath, val_output_size, 'val', args.guideL, args.guideR, normalize)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=2, shuffle=True,
                            num_workers=4, pin_memory=True, sampler=None,
                            worker_init_fn=lambda work_id: np.random.seed(work_id))
                            # worker_init_fn ensures different sampling patterns for
                            # each data loading thread
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False, pin_memory=True,
                        num_workers=4)

#normalize = info.load_vaihingen_info(args.info_datapath)

norm_mode = ['naive_categorical', # Applying categorical CBN on 3D-CNN in stereo matching network
                'naive_continuous', # Applying continuous CBN on 3D-CNN in stereo matching network
                'categorical', # Applying categorical CCVNorm on 3D-CNN in stereo matching network
                'continuous', # Applying continuous CCVNorm on 3D-CNN in stereo matching network
                'categorical_hier', # Applying categorical HierCCVNorm on 3D-CNN in stereo matching network
                ][4]

if args.model == 'basic':
    print('load basic model')
    model = PSMNetLiDAR(maxdisparity = args.maxdisp, norm_mode = norm_mode)
elif args.model == 'deux':
    print('load deux model')
    model = PSMNetLiDARDeux(maxdisparity = args.maxdisp, norm_mode = norm_mode)
elif args.model == 'guide':
    print('load guide model')
    model = PSMNetLiDARGuide(maxdisparity = args.maxdisp, norm_mode = norm_mode)
elif args.model == 'full':
    print('load guide full model')
    model = PSMNetLiDARGuideFull(maxdisparity = args.maxdisp, norm_mode = norm_mode)
elif args.model == 'basic2x':
    print('load basic2x model')
    model = PSMNetLiDAR2x(maxdisparity = args.maxdisp, norm_mode = norm_mode)
elif args.model == 'deux2x':
    print('load deux2x model')
    model = PSMNetLiDARDeux2x(maxdisparity = args.maxdisp, norm_mode = norm_mode)
else:
    print('no model')
#for name, layer in model.named_modules():
#    print(name, layer)

if args.cuda:
    model = nn.DataParallel(model) #device_ids=[0,1])
    model.cuda()

if args.loadmodel is not None:
    state_dict = torch.load(args.loadmodel)
    model.load_state_dict(state_dict['state_dict'])

    print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

    optimizer = optim.Adam(model.parameters(), lr=0.1, betas=(0.9, 0.999))
    
    # check point
    # resume problem
    if args.resume :
        # torch 0.4
        #optimizer = state_dict['optimizer']
        optimizer.load_state_dict(state_dict['optimizer'])
    # torch 0.4
    #else :
    #    optimizer = optim.Adam(model.parameters(), lr=0.1, betas=(0.9, 0.999))

else:
    optimizer = optim.Adam(model.parameters(), lr=0.1, betas=(0.9, 0.999))
    
def train(inputs, disp_true):
    # a transform for depth
    disp_true = torch.squeeze(disp_true,1)
    #---------
    #pdb.set_trace()
    mask1 = (disp_true > 0)
    mask2 = (disp_true < args.maxdisp)
    mask = mask1 & mask2
    #mask = torch.logical_and(mask1, mask2)
    mask.detach_()
    #----

    optimizer.zero_grad()
    
    output1, output2, output3 = model(inputs)

    output1 = torch.squeeze(output1,1)
    output2 = torch.squeeze(output2,1)
    output3 = torch.squeeze(output3,1)

    loss = 0.5*F.smooth_l1_loss(output1[mask], disp_true[mask], size_average=True) + 0.7*F.smooth_l1_loss(output2[mask], disp_true[mask], size_average=True) + F.smooth_l1_loss(output3[mask], disp_true[mask], size_average=True) 

    loss.backward()
    optimizer.step()

    #pdb.set_trace()
    
    # pytorch 0.41
    #return loss.data[0]
    return loss.data

def test(inputs, disp_true):
    # a transform for depth
    disp_true = torch.squeeze(disp_true,1)
    disp_true = disp_true.data.cpu()

    #computing 3-px error#
    true_disp = disp_true
    mask1 = (disp_true > 0)
    mask2 = (disp_true < args.maxdisp)
    mask = mask1 & mask2

    with torch.no_grad():
        output3 = model(inputs)
        output3 = torch.squeeze(output3, 1)

    pred_disp = output3.data.cpu()

    #pdb.set_trace()
    index = np.argwhere(mask)
    #print(index)
    #index = np.argwhere(true_disp > 0)
    # here is a bug(the test exmaple number should num%batch_size != 1)
    # too many indices for tensor of dimension 2
    disp_true[index[0][:], index[1][:], index[2][:]] = np.abs(true_disp[index[0][:], index[1][:], index[2][:]]-pred_disp[index[0][:], index[1][:], index[2][:]])
    correct = (disp_true[index[0][:], index[1][:], index[2][:]] < 3)|(disp_true[index[0][:], index[1][:], index[2][:]] < true_disp[index[0][:], index[1][:], index[2][:]]*0.05)      
    torch.cuda.empty_cache()

    # crop the image
    if len(index[0]) > 0:
        return 1-(float(torch.sum(correct))/float(len(index[0])))
    else :
        return 1

def adjust_learning_rate(optimizer, epoch):
    if epoch <= 200:
       lr = 0.001
    else:
       lr = 0.0001
    #print(lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def main():
    max_acc=0
    max_epo=0
    start_full_time = time.time()
    for epoch in range(args.epoch_start, args.epochs+1):
        total_train_loss = 0
        total_test_loss = 0
        adjust_learning_rate(optimizer,epoch)
        
        ## training ##
        model.train()

        for batch_idx, data in enumerate(train_loader):
            # Pack data
            if args.cuda:
                for k in data.keys():
                    data[k] = data[k].cuda()
            inputs = dict()
            inputs['left_rgb'] = data['left_rgb']
            inputs['right_rgb'] = data['right_rgb']
            inputs['left_sd'] = data['left_sd']
            inputs['right_sd'] = data['right_sd']
            target = data['left_d']

            start_time = time.time()

            #pdb.set_trace()
            loss = train(inputs, target)


            print('Iter %d training loss = %.3f , time = %.2f' %(batch_idx, loss, time.time() - start_time))
            total_train_loss += loss
        print('epoch %d total training loss = %.3f' %(epoch, total_train_loss/len(train_loader)))

        ## Test ##
        model.eval()
        for batch_idx, data in enumerate(val_loader):
            # Pack data
            if args.cuda:
                for k in data.keys():
                    data[k] = data[k].cuda()
            inputs = dict()
            inputs['left_rgb'] = data['left_rgb']
            inputs['right_rgb'] = data['right_rgb']
            inputs['left_sd'] = data['left_sd']
            inputs['right_sd'] = data['right_sd']
            target = data['left_d']

            test_loss = test(inputs, target)
            
            #print('Iter %d 3-px error in val = %.3f' %(batch_idx, test_loss*100))
            total_test_loss += test_loss


        print('epoch %d total 3-px error in val = %.3f' %(epoch, total_test_loss/len(val_loader)*100))
        if total_test_loss/len(val_loader)*100 > max_acc:
            max_acc = total_test_loss/len(val_loader)*100
            max_epo = epoch
        
        #print('MAX epoch %d total test error = %.3f' %(max_epo, max_acc))
        
        #SAVE
        if (epoch + 1) % 5==0 :
            savefilename = args.savemodel+'finetune_'+str(epoch)+'.tar'
            torch.save({
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'train_loss': total_train_loss/len(train_loader),
                    'test_loss': total_test_loss/len(val_loader)*100,
                }, savefilename)
	
        #print('full finetune time = %.2f HR' %((time.time() - start_full_time)/3600))
    print(max_epo)
    print(max_acc)


if __name__ == '__main__':
   main()
