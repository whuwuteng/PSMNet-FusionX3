
import os

from model.psmnet_lidar_2x import PSMNetLiDAR2x
from model.psmnet_lidar_guide_full import PSMNetLiDARGuideFull

from model.psmnet_lidar_deux import PSMNetLiDARDeux

from model.psmnet_lidar_guide_full_2x import PSMNetLiDARGuideFull2x

import pytorch_lightning as pl
from model.smoothloss import SmoothL1Loss
import torch.optim as optim
import torch
import torch.nn.functional as F
import numpy as np

import imageio

# like a lightning interface
class PSMNetLiDARWeight2x_lightning(pl.LightningModule):
    def __init__(self, max_disp=192, disp_scale = 256, norm_mode = 'categorical', learning_rate = 0.001):
        super().__init__()
        self.model = PSMNetLiDARGuideFull2x(max_disp, norm_mode = norm_mode)
        self.loss = SmoothL1Loss()
        self.max_disp = max_disp
        self.disp_scale = disp_scale
        self.lr = learning_rate

    def forward(self, inputs):
        return self.model(inputs)
    
    def configure_optimizers(self):
      optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
      return optimizer
    
    def training_step(self, batch, batch_idx):
        """left_img = batch['left']
        right_img = batch['right']
        target_disp = batch['disp']"""

        left_weight = batch['left_cmp']
        target_disp = batch['left_d']
        target_disp = torch.squeeze(target_disp, 1)
        left_weight = torch.squeeze(left_weight, 1)

        mask = (target_disp > 0 )& (target_disp < self.max_disp) 
        mask = mask.detach_()

        disp1, disp2, disp3 = self(batch)
        disp1 = torch.squeeze(disp1, 1)
        disp2 = torch.squeeze(disp2, 1)
        disp3 = torch.squeeze(disp3, 1)

        #loss1, loss2, loss3 = self.loss(disp1[mask], disp2[mask], disp3[mask], target_disp[mask])
        #total_loss = 0.5 * loss1 + 0.7 * loss2 + 1.0 * loss3
        """loss1 = 0
        loss2 = 0
        loss3 = 0
        valid_pixels = np.argwhere(mask.cpu())
        for pixel in valid_pixels :
            pixel_pred1 = disp1[pixel[0], pixel[1]]
            pixel_pred2 = disp2[pixel[0], pixel[1]]
            pixel_pred3 = disp3[pixel[0], pixel[1]]
            pixel_target = target_disp[pixel[0], pixel[1]]
            pixel_weight = left_weight[pixel[0], pixel[1]]*5.0

            loss1 = loss1 + pixel_weight*F.smooth_l1_loss(pixel_pred1, pixel_target)
            loss2 = loss2 + pixel_weight*F.smooth_l1_loss(pixel_pred2, pixel_target)
            loss3 = loss3 + pixel_weight*F.smooth_l1_loss(pixel_pred3, pixel_target)"""
        
        # refer to https://stackoverflow.com/questions/60252902/implementing-smoothl1loss-for-specific-case not work
        # refer to https://discuss.pytorch.org/t/unet-pixel-wise-weighted-loss-function/46689/3
        loss1 = F.smooth_l1_loss(disp1[mask], target_disp[mask], reduction='none')
        loss1 = torch.mean(loss1 * left_weight[mask])
        loss2 = F.smooth_l1_loss(disp2[mask], target_disp[mask], reduction='none')
        loss2 = torch.mean(loss2 * left_weight[mask])
        loss3 = F.smooth_l1_loss(disp3[mask], target_disp[mask], reduction='none')
        loss3 = torch.mean(loss3 * left_weight[mask])

        """loss1 = F.smooth_l1_loss(disp1[mask], target_disp[mask])
        loss2 = F.smooth_l1_loss(disp2[mask], target_disp[mask])
        loss3 = F.smooth_l1_loss(disp3[mask], target_disp[mask])"""

        total_loss = 0.5 * loss1 + 0.7 * loss2 + 1.0 * loss3
        #self.log('training_loss', total_loss, on_step=True, on_epoch=False, prog_bar=True)
        print('training_loss: ', total_loss)

        return total_loss

    def training_epoch_start(self):
        print("current lr: ", self.lr)
    
    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx, optimizer_closure, on_tpu, using_native_amp, using_lbfgs):
        if epoch == 200:
            self.lr = 0.0001
            for pg in optimizer.param_groups:
                pg["lr"] = self.lr
        # update params
        optimizer.step(closure=optimizer_closure)
    
            
    def validation_step(self, batch, batch_idx):
        avg_error = 0.0

        """left_img = batch['left']
        right_img = batch['right']"""

        target_disp = batch['left_d']
        target_disp = torch.squeeze(target_disp, 1)

        #print('*****DEBUG********')
        #print(target_disp.size())

        mask = (target_disp > 0 )& (target_disp < self.max_disp) 
        mask = mask.detach_()

        #disp = self(left_img, right_img)
        disp = self(batch)
        disp = torch.squeeze(disp, 1)

        #calculate 3px error
        #print('disp: ', disp)
        #print('target: ', target_disp)

        delta = torch.abs(disp[mask] - target_disp[mask])
        error_mat = (((delta >= 3.0) + (delta >= 0.05 * (target_disp[mask]))) == 2)

        if torch.numel(disp[mask]) > 0:
            error = torch.sum(error_mat).item() / torch.numel(disp[mask]) * 100
        else :
            return 0

        """index = np.argwhere(mask)

        disp_true[index[0][:], index[1][:], index[2][:]] = np.abs(true_disp[index[0][:], index[1][:], index[2][:]]-pred_disp[index[0][:], index[1][:], index[2][:]])
        correct = (disp_true[index[0][:], index[1][:], index[2][:]] < 3)|(disp_true[index[0][:], index[1][:], index[2][:]] < true_disp[index[0][:], index[1][:], index[2][:]]*0.05)

        error = 1-(float(torch.sum(correct))/float(len(index[0])))"""

        # cuda
        #print('evaluation error: ', error)
        return error


    def validation_epoch_end(self, validation_step_outputs):
        mean_error = sum(validation_step_outputs) / len(validation_step_outputs)
        #self.log('3px_error', mean_error, on_step=False, on_epoch=True, prog_bar=True)
    
    # https://github.com/Lightning-AI/lightning/issues/1088
    def test_step(self, batch, batch_idx):
        # OPTIONAL
        """left_img = batch['left']
        right_img = batch['right']"""

        # just the name
        # ?
        target_disp = batch['disp']

        disp = self(batch)
        disp = torch.squeeze(disp, 1)
        disp = torch.squeeze(disp)

        # save the result
        pred_disp = disp.data.cpu().numpy()

        img = (pred_disp*self.disp_scale).astype('uint16')

        #print('***********: ', target_disp[0])

        save_path, filename = os.path.split(target_disp[0])
        if not os.path.exists(save_path) :
            os.makedirs(save_path)
                
        #print('save: ' + target_disp[0])
        #print('***************: ', img)
        imageio.imwrite(target_disp[0], img)

        """print('***************over***************')
        return 0

    def test_epoch_end(self, test_step_outputs):
        print('***************end***************')
        return {'avg_test_loss': 0, 'avg_acc': 0, 'log': 0}"""