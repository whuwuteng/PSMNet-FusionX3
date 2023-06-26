from __future__ import print_function
import argparse
import os
import sys
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import time
import math
from dataloader import listKITTIfiles as listLoader
from dataloader import KITTILoader as dataLoader
import cv2
from dataloader import vaihingen_collector_file as vse
from models import *
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
parser = argparse.ArgumentParser(description='Guided-Stereo')
parser.add_argument('--datapath', default='2011_09_26_0011/',
                    help='datapath')
parser.add_argument('--maxdisp', type=int ,default=192,
                    help='maxium disparity')
parser.add_argument('--loadmodel', default=None,
                    help='load model')
parser.add_argument('--train_datapath', default=None,
                    help='training data path')
parser.add_argument('--val_datapath', default=None,
                    help='validation data path')
parser.add_argument('--disp_scale', type=int ,default=256,
                    help='maxium disparity')
parser.add_argument('--resume', action='store_true', default=False,
                    help='load model and optimizer')
parser.add_argument('--epoch_start', type=int, default=0, help='number of epochs to train for')               
parser.add_argument('--epochs', type=int, default=300,
                    help='number of epochs to train')
parser.add_argument('--savemodel', default='./',
                    help='save model')
parser.add_argument('--folder', default='guide',
                    help='guide disparity path')                    
parser.add_argument('--guided', action='store_true', default=False, help='Enable guided stereo')
parser.add_argument('--display', action='store_true', default=False, help='Display output')
parser.add_argument('--save', action='store_true', default=False, help='Save output')
parser.add_argument('--verbose', action='store_true', default=False, help='Print stats for each single image')
parser.add_argument('--no-cuda', action='store_true', default=False, help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')

args = parser.parse_args()
print('[***] args= ', args)

args.cuda = not args.no_cuda and torch.cuda.is_available()

#all_left, all_right, all_guide, all_disp = listLoader.dataloader(args.train_datapath)
#test_left, test_right, test_guide, test_disp = listLoader.dataloader(args.val_datapath)

all_left, all_right, all_guide, all_disp, test_left, test_right, test_guide, test_disp = vse.datacollectorall(
    args.train_datapath, args.val_datapath, args.folder)

#print(test_left)

# read training
TrainImgLoader = torch.utils.data.DataLoader(
         dataLoader.imageLoader(all_left,all_right,all_guide, all_disp, True), 
         batch_size= 12, shuffle= True, num_workers= 6, drop_last=False)

TestImgLoader = torch.utils.data.DataLoader(
         dataLoader.imageLoader(test_left, test_right, test_guide, test_disp, False), 
         batch_size= 4, shuffle= False, num_workers= 4, drop_last=False)

# build model
model = psmnet(args.maxdisp, args.guided)

if args.cuda:
    model = nn.DataParallel(model)
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
 
def train(reference,imgL,imgR,guideL,disp_true,h,w):
        model.train()
        #pdb.set_trace()
        imgL   = Variable(torch.FloatTensor(imgL))
        imgR   = Variable(torch.FloatTensor(imgR))
        validhints = (guideL > 0).float()

        #disp_L = Variable(torch.FloatTensor(disp_L))

        if args.cuda:
            #imgL, imgR = imgL.cuda(), imgR.cuda()
            imgL, imgR, disp_true = imgL.cuda(), imgR.cuda(), disp_true.cuda()

        #---------
        #pdb.set_trace()
        mask1 = (disp_true > 0)
        mask2 = (disp_true < args.maxdisp)
        mask = mask1 & mask2
        #mask = torch.logical_and(mask1, mask2)
        mask.detach_()
        #print(mask)
        #----

        optimizer.zero_grad()
    
        output1, output2, output3 = model(imgL,imgR,guideL,validhints,k=10,c=1)
        output1 = torch.squeeze(output1,1)
        output2 = torch.squeeze(output2,1)
        output3 = torch.squeeze(output3,1)
        #loss ?
        loss = 0.5*F.smooth_l1_loss(output1[mask], disp_true[mask], size_average=True) + 0.7*F.smooth_l1_loss(output2[mask], disp_true[mask], size_average=True) + F.smooth_l1_loss(output3[mask], disp_true[mask], size_average=True) 

        loss.backward()
        
        optimizer.step()

        #pdb.set_trace()
        
        # pytorch 0.41
        #return loss.data[0]
        return loss.data

# Running guided stereo!
def test(reference,imgL,imgR,guideL,disp_true,h,w,batch_idx):
        model.eval()
        imgL   = Variable(torch.FloatTensor(imgL))
        imgR   = Variable(torch.FloatTensor(imgR))   

        validhints = (guideL > 0).float()

        #computing density
        density=(float(torch.nonzero(validhints).size(0)) / ((validhints.size(1))*validhints.size(2))*100.)

        if args.cuda:
            imgL, imgR = imgL.cuda(), imgR.cuda()

        with torch.no_grad():
            output3 = model(imgL,imgR,guideL,validhints,k=10,c=1)

        top_pad   = 384-h
        left_pad  = 1280-w

        pred_disp = output3.data.cpu()
        if args.display or args.save:
            display_and_save(batch_idx, reference*255, guideL, pred_disp, disp_true, top_pad, left_pad)

        true_disp_nog = disp_true.clone()
        true_disp_all = disp_true.clone()

        # compute NoG error
        # bad 2 for no guided pixel 
        index_nog = np.argwhere(true_disp_nog*(1-validhints)>0)
        true_disp_nog[index_nog[0][:], index_nog[1][:], index_nog[2][:]] = np.abs(true_disp_nog[index_nog[0][:], index_nog[1][:], index_nog[2][:]]-pred_disp[index_nog[0][:], index_nog[1][:], index_nog[2][:]])
        correct2_nog = (true_disp_nog[index_nog[0][:], index_nog[1][:], index_nog[2][:]] < 2)
        bad2_nog = 1-(float(torch.sum(correct2_nog))/float(len(index_nog[0])))
        avg_nog  = float(torch.sum(true_disp_nog[index_nog[0][:], index_nog[1][:], index_nog[2][:]])/float(len(index_nog[0])))

        # compute All error
        # bad 2 for all the pixel
        index_all = np.argwhere(true_disp_all>0)
        true_disp_all[index_all[0][:], index_all[1][:], index_all[2][:]] = np.abs(true_disp_all[index_all[0][:], index_all[1][:], index_all[2][:]]-pred_disp[index_all[0][:], index_all[1][:], index_all[2][:]])
        correct2_all = (true_disp_all[index_all[0][:], index_all[1][:], index_all[2][:]] < 2)
        bad2_all = 1-(float(torch.sum(correct2_all))/float(len(index_all[0])))
        avg_all  = float(torch.sum(true_disp_all[index_all[0][:], index_all[1][:], index_all[2][:]])/float(len(index_all[0])))

        return bad2_all, bad2_nog, avg_all, avg_nog, density

# Dirty work to show/save results...
def display_and_save(batch_idx, left, guide, disparity, gt, top_pad, left_pad):
        left_2show = np.transpose(left.cpu().numpy()[0][:,top_pad:,left_pad:], (1,2,0)).astype(np.uint8)
        left_2show = cv2.cvtColor(left_2show, cv2.COLOR_BGR2RGB)
        disp_2show = cv2.applyColorMap(np.clip(50+2*disparity.numpy()[0][top_pad:,left_pad:], a_min=0, a_max=255.).astype(np.uint8), cv2.COLORMAP_JET)
        guide_2show = cv2.applyColorMap(np.clip(50+2*guide.numpy()[0][top_pad:,left_pad:], a_min=0, a_max=255.).astype(np.uint8), cv2.COLORMAP_JET) * np.expand_dims((guide.numpy()[0][top_pad:,left_pad:]>0),-1)
        gt_2show = cv2.applyColorMap(np.clip(50+2*gt.numpy()[0][top_pad:,left_pad:], a_min=0, a_max=255.).astype(np.uint8), cv2.COLORMAP_JET)* np.expand_dims((gt.numpy()[0][top_pad:,left_pad:]>0),-1)

        collage = np.concatenate((np.concatenate((left_2show, guide_2show), 1), np.concatenate((disp_2show, gt_2show), 1)), 0)
        collage = cv2.resize(collage, (collage.shape[1]//2, collage.shape[0]//2))
        if args.display:
            cv2.imshow("Guided stereo", collage)
            k = cv2.waitKey(10)
            if k == 27: 
                sys.exit()  # esc to quit
        if args.save:
            cv2.imwrite(args.output_dir+"/%06d.png"%batch_idx, collage)

def adjust_learning_rate(optimizer, epoch):
    if epoch <= 200:
       lr = 0.001
    else:
       lr = 0.0001
    print(lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

# main
def main():
    max_acc = 0
    max_epo = 0
    start_full_time = time.time()
    
    for epoch in range(args.epoch_start, args.epochs+1):
        total_train_loss = 0
        adjust_learning_rate(optimizer,epoch)
        
        ## training ##
        for batch_idx, (reference, imgL, imgR, guideL, dispL, h, w) in enumerate(TrainImgLoader):
            start_time = time.time()
            #pdb.set_trace()
            loss = train(reference,imgL,imgR,guideL,dispL,h,w)
            
            print('Iter %d training loss = %.3f , time = %.2f' %(batch_idx, loss, time.time() - start_time))
            total_train_loss += loss
        print('epoch %d total training loss = %.3f' %(epoch, total_train_loss/len(TrainImgLoader)))

        total_bad2_all = 0
        total_bad2_nog = 0
        total_avg_all = 0
        total_avg_nog = 0
        total_density = 0

        ## Test ##
        if args.verbose:
            print('Frame & bad2-all & bad2-NoG & MAE-all & MAE-NoG & density')
        
        for batch_idx, (reference, imgL, imgR, guideL, dispL, h, w) in enumerate(TestImgLoader):
            bad2_all, bad2_nog, avg_all, avg_nog, density = test(reference,imgL,imgR,guideL,dispL,h,w,batch_idx)
            total_bad2_all += bad2_all
            total_bad2_nog += bad2_nog
            total_avg_all  += avg_all
            total_avg_nog  += avg_nog
            total_density += density
            
            if args.verbose:
                print('%06d & %.2f & %.2f & %.2f & %.2f & %.2f' %(batch_idx, bad2_all*100., bad2_nog*100., avg_all, avg_nog, density))

        print("bad2-all & bad2-NoG & MAE-all & MAE-NoG & density")
        print('%.2f & %.2f & %.2f & %.2f & %.2f' % ((total_bad2_all/len(TestImgLoader)*100), (total_bad2_nog/len(TestImgLoader)*100), (total_avg_all/len(TestImgLoader)), (total_avg_nog/len(TestImgLoader)), (total_density/len(TestImgLoader))))

        if total_bad2_all/len(TestImgLoader)*100 > max_acc:
            max_acc = total_bad2_all/len(TestImgLoader)*100
            max_epo = epoch
        
        print('MAX epoch %d total test error = %.3f' %(max_epo, max_acc))
        
        #SAVE
        if (epoch + 1) % 5==0 :
            savefilename = args.savemodel+'finetune_'+str(epoch)+'.tar'
            torch.save({
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'train_loss': total_train_loss/len(TrainImgLoader),
                    'test_loss': total_bad2_all/len(TestImgLoader)*100,
                }, savefilename)
	
        print('full finetune time = %.2f HR' %((time.time() - start_full_time)/3600))
    print(max_epo)
    print(max_acc)


if __name__ == '__main__':
   main()
