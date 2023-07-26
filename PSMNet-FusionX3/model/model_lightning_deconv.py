
import os

from model.stackhourglass_deconv_2x import PSMNetDeConv2x
import pytorch_lightning as pl
from model.smoothloss import SmoothL1Loss
import torch.optim as optim
import torch
#import numpy as np

import imageio

# like a lightning interface
class PSMNetDeconv_lightning(pl.LightningModule):
    def __init__(self, max_disp=192, disp_scale = 256, learning_rate = 0.001):
        super().__init__()
        self.model = PSMNetDeConv2x(max_disp)
        self.loss = SmoothL1Loss()
        self.max_disp = max_disp
        self.disp_scale = disp_scale
        self.lr = learning_rate

    def forward(self, left, right):
        return self.model(left, right)
    
    def configure_optimizers(self):
      optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
      return optimizer
    
    def training_step(self, batch, batch_idx):
        left_img = batch['left']
        right_img = batch['right']
        target_disp = batch['disp']

        mask = (target_disp > 0 )& (target_disp < self.max_disp) 
        mask = mask.detach_()

        disp1, disp2, disp3 = self(left_img, right_img)
        disp1 = torch.squeeze(disp1, 1)
        disp2 = torch.squeeze(disp2, 1)
        disp3 = torch.squeeze(disp3, 1)

        loss1, loss2, loss3 = self.loss(disp1[mask], disp2[mask], disp3[mask], target_disp[mask])
        total_loss = 0.5 * loss1 + 0.7 * loss2 + 1.0 * loss3

        #self.log('training_loss', total_loss, on_step=True, on_epoch=False, prog_bar=True)

        return total_loss

    def training_epoch_start(self):
        print("current lr: ", self.lr)
    
    # refer to https://stackoverflow.com/questions/71236391/pytorch-lightning-print-accuracy-and-loss-at-the-end-of-each-epoch
    # and https://stackoverflow.com/questions/38286718/what-does-def-main-none-do
    def training_epoch_end(self, outputs):
        gathered = self.all_gather(outputs)
        if self.global_rank == 0:
            # print(gathered)
            loss = sum(output['loss'].mean() for output in gathered) / len(outputs)
            print("training loss: ",loss.item())

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx, optimizer_closure, on_tpu, using_native_amp, using_lbfgs):
        if epoch == 200:
            self.lr = 0.0001
            for pg in optimizer.param_groups:
                pg["lr"] = self.lr
        # update params
        optimizer.step(closure=optimizer_closure)
    
            
    def validation_step(self, batch, batch_idx):
        avg_error = 0.0

        left_img = batch['left']
        right_img = batch['right']
        target_disp = batch['disp']

        mask = (target_disp > 0 )& (target_disp < self.max_disp) 
        mask = mask.detach_()

        #_, _, disp = self(left_img, right_img)
        disp = self(left_img, right_img)
        disp = torch.squeeze(disp, 1)

        #calculate 3px error
        #print('disp: ', disp)
        #print('target: ', target_disp)

        delta = torch.abs(disp[mask] - target_disp[mask])
        error_mat = (((delta >= 3.0).long() + (delta >= 0.05 * (target_disp[mask])).long()) == 2)

        if torch.numel(disp[mask]) > 0:
            error = float(torch.sum(error_mat).item()) / float(torch.numel(disp[mask])) * 100
        else :
            return 0

        """index = np.argwhere(mask)

        disp_true[index[0][:], index[1][:], index[2][:]] = np.abs(true_disp[index[0][:], index[1][:], index[2][:]]-pred_disp[index[0][:], index[1][:], index[2][:]])
        correct = (disp_true[index[0][:], index[1][:], index[2][:]] < 3)|(disp_true[index[0][:], index[1][:], index[2][:]] < true_disp[index[0][:], index[1][:], index[2][:]]*0.05)

        error = 1-(float(torch.sum(correct))/float(len(index[0])))"""

        return error

    # refer to https://stackoverflow.com/questions/66290662/how-to-get-step-wise-validation-loss-curve-over-all-epochs-in-pytorch-lightning
    def validation_epoch_end(self, outputs):
        # different node
        """mean_error = sum(outputs) / len(outputs)
        #self.log('3px_error', mean_error, on_step=False, on_epoch=True, prog_bar=True)
        print("testing loss: ", mean_error)"""
        gathered = self.all_gather(outputs)
        if self.global_rank == 0:
            # print(gathered)
            loss = sum(outputs.mean() for outputs in gathered) / len(outputs)

        print("testing loss: ", loss.item())
    
    # https://github.com/Lightning-AI/lightning/issues/1088
    def test_step(self, batch, batch_idx):
        # OPTIONAL
        left_img = batch['left']
        right_img = batch['right']

        # just the name
        target_disp = batch['disp']

        disp = self(left_img, right_img)
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