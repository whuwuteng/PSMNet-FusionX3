"""
Stereo + LiDAR fusion: incorporate sparse disparity map into stereo matching network.
"""

import torch
import torch.nn as nn
from torch.autograd import Variable

from .gcnet_conv import net_init, conv2d_bn, conv_res, conv3d_bn, deconv3d_bn, conv3d_ccvnorm, deconv3d_ccvnorm
from .gcnet_fun import myAdd3d

# PSMNet
from .submodule_2x import feature_extraction2x, disparityregression

flag_bias_t = True
flag_bn = True
activefun_t = nn.ReLU(inplace=True)

# only ccvnorm
class PSMNetLiDAR2x(nn.Module):
    def __init__(self, maxdisparity=192, num_F=32, channel=3, norm_mode='categorical'):
        super(PSMNetLiDAR2x, self).__init__()
        assert norm_mode in ['naive_categorical', 'naive_continuous', 'categorical', 'continuous', 'categorical_hier']
        self.maxdisp = maxdisparity
        self.norm_mode = norm_mode
        self.D = maxdisparity // 2
        self.F = num_F
  
        self.count_levels = 1
        self.layer2d = feature_extraction2x(channel)

        if 'continuous' in self.norm_mode:
            self.down_2x = nn.MaxPool2d(2)
        elif 'categorical' in self.norm_mode:
            self.down_2x = nn.MaxPool3d(2)
        else:
            raise NotImplementedError

        # only the HierCCVNorm is used
        self.dres01 = conv3d_bn(self.F*2, self.F, kernel_size=3, stride=1, flag_bias=flag_bias_t, bn=flag_bn, activefun=activefun_t)

        # add ReLU
        self.dres02 = conv3d_bn(self.F,   self.F, kernel_size=3, stride=1, flag_bias=flag_bias_t, bn=flag_bn, activefun=activefun_t)
        # add ReLU

        self.dres11 = conv3d_bn(self.F,   self.F, kernel_size=3, stride=1, flag_bias=flag_bias_t, bn=flag_bn, activefun=activefun_t)
        # add ReLU

        self.dres12 = conv3d_bn(self.F,   self.F, kernel_size=3, stride=1, flag_bias=flag_bias_t, bn=flag_bn, activefun=None)

        #self.dres21 = conv3d_bn(self.F,   self.F * 2, kernel_size=3, stride=2, flag_bias=flag_bias_t, bn=flag_bn, activefun=activefun_t)
        # add ReLU
        
        self.dres21 = conv3d_ccvnorm(self.F, self.F * 2, self.D//2, kernel_size=3, stride=2, flag_bias=flag_bias_t, bn=flag_bn, 
                                  activefun=activefun_t, mode=self.norm_mode, norm_in_channels=32)

        self.dres22 = conv3d_bn(self.F * 2,  self.F * 2, kernel_size=3, stride=1, flag_bias=flag_bias_t, bn=flag_bn, activefun=None)

        #self.dres23 = conv3d_bn(self.F * 2,   self.F * 2, kernel_size=3, stride=2, flag_bias=flag_bias_t, bn=flag_bn, activefun=activefun_t)
        # add ReLU

        self.dres23 = conv3d_ccvnorm(self.F * 2, self.F * 2, self.D//4, kernel_size=3, stride=2, flag_bias=flag_bias_t, bn=flag_bn, 
                                  activefun=activefun_t, mode=self.norm_mode, norm_in_channels=64)

        self.dres24 = conv3d_bn(self.F * 2,   self.F * 2, kernel_size=3, stride=1, flag_bias=flag_bias_t, bn=flag_bn, activefun=activefun_t)
        # add ReLU

        #self.dres25 = deconv3d_bn(self.F*2, self.F * 2, kernel_size=3, stride=2, flag_bias=flag_bias_t, bn=flag_bn, activefun=None)

        self.dres25 = deconv3d_ccvnorm(self.F*2, self.F*2, self.D//2, kernel_size=3, stride=2, flag_bias=flag_bias_t, bn=flag_bn, 
                                    activefun=activefun_t, mode=self.norm_mode)
        
        #self.dres26 = deconv3d_bn(self.F*2, self.F, kernel_size=3, stride=2, flag_bias=flag_bias_t, bn=flag_bn, activefun=None)

        self.dres26 = deconv3d_ccvnorm(self.F*2, self.F, self.D, kernel_size=3, stride=2, flag_bias=flag_bias_t, bn=flag_bn, 
                                    activefun=activefun_t, mode=self.norm_mode)


        #self.dres31 = conv3d_bn(self.F,   self.F * 2, kernel_size=3, stride=2, flag_bias=flag_bias_t, bn=flag_bn, activefun=activefun_t)
        # add ReLU
        
        self.dres31 = conv3d_ccvnorm(self.F, self.F * 2, self.D//2, kernel_size=3, stride=2, flag_bias=flag_bias_t, bn=flag_bn, 
                                  activefun=activefun_t, mode=self.norm_mode, norm_in_channels=32)

        self.dres32 = conv3d_bn(self.F * 2,   self.F * 2, kernel_size=3, stride=1, flag_bias=flag_bias_t, bn=flag_bn, activefun=None)

        #self.dres33 = conv3d_bn(self.F * 2,   self.F * 2, kernel_size=3, stride=2, flag_bias=flag_bias_t, bn=flag_bn, activefun=activefun_t)
        # add ReLU
        self.dres33 = conv3d_ccvnorm(self.F * 2, self.F * 2, self.D//4, kernel_size=3, stride=2, flag_bias=flag_bias_t, bn=flag_bn, 
                                  activefun=activefun_t, mode=self.norm_mode, norm_in_channels=64)

        self.dres34 = conv3d_bn(self.F * 2,  self.F * 2, kernel_size=3, stride=1, flag_bias=flag_bias_t, bn=flag_bn, activefun=activefun_t)
        # add ReLU

        #self.dres35 = deconv3d_bn(self.F*2, self.F * 2, kernel_size=3, stride=2, flag_bias=flag_bias_t, bn=flag_bn, activefun=None)

        self.dres35 = deconv3d_ccvnorm(self.F*2, self.F*2, self.D//2, kernel_size=3, stride=2, flag_bias=flag_bias_t, bn=flag_bn, 
                                    activefun=activefun_t, mode=self.norm_mode)
        
        #self.dres36 = deconv3d_bn(self.F*2, self.F, kernel_size=3, stride=2, flag_bias=flag_bias_t, bn=flag_bn, activefun=None)

        self.dres36 = deconv3d_ccvnorm(self.F*2, self.F, self.D, kernel_size=3, stride=2, flag_bias=flag_bias_t, bn=flag_bn, 
                                    activefun=activefun_t, mode=self.norm_mode)

        #self.dres41 = conv3d_bn(self.F,   self.F * 2, kernel_size=3, stride=2, flag_bias=flag_bias_t, bn=flag_bn, activefun=activefun_t)
        # add ReLU

        self.dres41 = conv3d_ccvnorm(self.F, self.F * 2, self.D//2, kernel_size=3, stride=2, flag_bias=flag_bias_t, bn=flag_bn, 
                                  activefun=activefun_t, mode=self.norm_mode, norm_in_channels=32)

        self.dres42 = conv3d_bn(self.F * 2,   self.F * 2, kernel_size=3, stride=1, flag_bias=flag_bias_t, bn=flag_bn, activefun=None)

        #self.dres43 = conv3d_bn(self.F * 2,   self.F * 2, kernel_size=3, stride=2, flag_bias=flag_bias_t, bn=flag_bn, activefun=activefun_t)
        # add ReLU

        self.dres43 = conv3d_ccvnorm(self.F * 2, self.F * 2, self.D//4, kernel_size=3, stride=2, flag_bias=flag_bias_t, bn=flag_bn, 
                                  activefun=activefun_t, mode=self.norm_mode, norm_in_channels=64)

        self.dres44 = conv3d_bn(self.F * 2,   self.F * 2, kernel_size=3, stride=1, flag_bias=flag_bias_t, bn=flag_bn, activefun=activefun_t)
        # add ReLU

        #self.dres45 = deconv3d_bn(self.F*2, self.F * 2, kernel_size=3, stride=2, flag_bias=flag_bias_t, bn=flag_bn, activefun=None)

        self.dres45 = deconv3d_ccvnorm(self.F*2, self.F*2, self.D//2, kernel_size=3, stride=2, flag_bias=flag_bias_t, bn=flag_bn, 
                                    activefun=activefun_t, mode=self.norm_mode)
        
        #self.dres46 = deconv3d_bn(self.F*2, self.F, kernel_size=3, stride=2, flag_bias=flag_bias_t, bn=flag_bn, activefun=None)

        self.dres46 = deconv3d_ccvnorm(self.F*2, self.F, self.D, kernel_size=3, stride=2, flag_bias=flag_bias_t, bn=flag_bn, 
                                    activefun=activefun_t, mode=self.norm_mode)

        self.classif11 =  conv3d_bn(self.F,   self.F, kernel_size=3, stride=1, flag_bias=flag_bias_t, bn=flag_bn, activefun=activefun_t)
        # add ReLU

        self.classif12 =  conv3d_bn(self.F, 1, kernel_size=3, stride=1, flag_bias=flag_bias_t, bn=flag_bn, activefun=None)

        self.classif21 =  conv3d_bn(self.F,   self.F, kernel_size=3, stride=1, flag_bias=flag_bias_t, bn=flag_bn, activefun=activefun_t)
        # add ReLU

        self.classif22 =  conv3d_bn(self.F, 1, kernel_size=3, stride=1, flag_bias=flag_bias_t, bn=flag_bn, activefun=None)

        self.classif31 =  conv3d_bn(self.F,   self.F, kernel_size=3, stride=1, flag_bias=flag_bias_t, bn=flag_bn, activefun=activefun_t)
        # add ReLU

        self.classif32 =  conv3d_bn(self.F, 1, kernel_size=3, stride=1, flag_bias=flag_bias_t, bn=flag_bn, activefun=None)

        # initial, same with net_init
        """for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1]*m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()"""

        net_init(self)

    def forward(self, inputs, mode='train'):
        imL, imR = inputs['left_rgb'], inputs['right_rgb']
        sdL, sdR = inputs['left_sd'], inputs['right_sd']
        assert imL.shape == imR.shape
        # Extract 2D features for left and right images (Input Fusion)

        refimg_fea = self.layer2d(imL)
        targetimg_fea = self.layer2d(imR)

         #matching
        cost = Variable(torch.FloatTensor(refimg_fea.size()[0], refimg_fea.size()[1]*2, self.maxdisp//2,  refimg_fea.size()[2],  refimg_fea.size()[3]).zero_()).cuda()

        #divide 4
        for i in range(self.maxdisp//2):
            if i > 0 :
             cost[:, :refimg_fea.size()[1], i, :,i:]   = refimg_fea[:,:,:,i:]
             cost[:, refimg_fea.size()[1]:, i, :,i:] = targetimg_fea[:,:,:,:-i]
            else:
             cost[:, :refimg_fea.size()[1], i, :,:]   = refimg_fea
             cost[:, refimg_fea.size()[1]:, i, :,:]   = targetimg_fea
        cost = cost.contiguous()

        # no guide 
        
        if not self.training:
            del refimg_fea,targetimg_fea
            torch.cuda.empty_cache()

        #print('mode: ')
        #print(self.norm_mode)

        if 'categorical' in self.norm_mode:
            mask = self.discretize_disp(sdL, self.maxdisp)
        elif 'continuous' in self.norm_mode:
            mask = sdL
        else:
            raise NotImplementedError

        mask_down2x = self.down_2x(mask)
        mask_down4x = self.down_2x(mask_down2x)
        mask_down8x = self.down_2x(mask_down4x)
        mask_down16x = self.down_2x(mask_down8x)
        mask_down32x = self.down_2x(mask_down16x)

        # combine the LiDAR
        # layer 1
        cost0 = self.dres01(cost)
        if not self.training:
            del cost
            torch.cuda.empty_cache()

        cost0 = self.dres02(cost0)

        # layer 2
        cost1 = self.dres11(cost0)
        cost1 = self.dres12(cost1)
        cost1 = cost1 + cost0
        if not self.training:
            del cost0
            torch.cuda.empty_cache()

        # layer 3?
        #print(cost1.size())
        #print(mask_down4x.size())
        out1 = self.dres21(cost1, mask_down4x)
        pre1 = self.dres22(out1)
        pre1 = nn.functional.relu(pre1, inplace=True)
        out1 = self.dres23(pre1, mask_down8x)
        out1 = self.dres24(out1)
        out1 = self.dres25(out1, mask_down4x) + pre1
        post1 = nn.functional.relu(out1, inplace=True)
        out1 = self.dres26(out1, mask_down2x)
        out1 = out1 + cost1

        # layer 4
        out2 = self.dres31(out1, mask_down4x)
        pre2 = self.dres32(out2)
        pre2 = nn.functional.relu(pre2 + post1, inplace=True)
        out2 = self.dres33(pre2, mask_down8x)
        out2 = self.dres34(out2)
        out2 = self.dres35(out2, mask_down4x) + pre1
        if not self.training:
            del pre1,post1
            torch.cuda.empty_cache()
            
        post2 = nn.functional.relu(out2, inplace=True)
        out2 = self.dres36(out2, mask_down2x)
        out2 = out2 + cost1

        # layer 5
        out3 = self.dres41(out2, mask_down4x)
        pre3 = self.dres42(out3)
        pre3 = nn.functional.relu(pre3 + post2, inplace=True)
        out3 = self.dres43(pre3, mask_down8x)
        out3 = self.dres44(out3)
        out3 = self.dres45(out3, mask_down4x) + pre2
        if not self.training:
            del pre2,post2
            torch.cuda.empty_cache()

        post3 = nn.functional.relu(out3, inplace=True)
        out3 = self.dres46(out3, mask_down2x)
        out3 = out3 + cost1
        if not self.training:
            del pre3,post3
            torch.cuda.empty_cache()

        # last step 
        cost1 = self.classif11(out1)
        cost1 = self.classif12(out1)
        cost2 = self.classif21(out2)
        cost2 = self.classif22(cost2) + cost1
        cost3 = self.classif31(out3)
        cost3 = self.classif32(out3) + cost2
        if not self.training:
            del out1,out2,out3
            torch.cuda.empty_cache()

        if self.training:
            cost1 = nn.functional.upsample(cost1, [self.maxdisp,imL.size()[2],imL.size()[3]], mode='trilinear')
            cost2 = nn.functional.upsample(cost2, [self.maxdisp,imL.size()[2],imL.size()[3]], mode='trilinear')

            cost1 = torch.squeeze(cost1,1)
            pred1 = nn.functional.softmax(cost1,dim=1)
            pred1 = disparityregression(self.maxdisp)(pred1)

            cost2 = torch.squeeze(cost2,1)
            pred2 = nn.functional.softmax(cost2,dim=1)
            pred2 = disparityregression(self.maxdisp)(pred2)

        cost3 = nn.functional.upsample(cost3, [self.maxdisp,imL.size()[2],imL.size()[3]], mode='trilinear')
        cost3 = torch.squeeze(cost3,1)
        pred3 = nn.functional.softmax(cost3,dim=1)

        if not self.training:
            del cost1, cost2, cost3
            torch.cuda.empty_cache()

        #For your information: This formulation 'softmax(c)' learned "similarity" 
        #while 'softmax(-c)' learned 'matching cost' as mentioned in the paper.
        #However, 'c' or '-c' do not affect the performance because feature-based cost volume provided flexibility.
        
        pred3 = disparityregression(self.maxdisp)(pred3)

        if self.training:
            return pred1, pred2, pred3
        else:
            return pred3

    # for the non value
    def discretize_disp(self, x, n_level):
        """ Discretize disparity: (n, 1, h, w) --> (n, n_level, h, w) 
            NOTE: for invalid point, set all to -1 (WARNING different from the previous, it's -1 not 1) """
        invalid_mask = (x <= 0).float() # NOTE: assuming x is sd --> use <= 0 for condition
        # NOTE: (1) multiplied by 2 because self.D = max_disp//2.
        #       (2) +/- 0.5 because the disparity level is centered at integer (e.g. 0,1,2...max_disp))
        lower = (torch.arange(0, n_level).float()[None, :, None, None].to(x) - 0.5) * 2
        upper = (torch.arange(0, n_level).float()[None, :, None, None].to(x) + 0.5) * 2
        disc_x = ((x.repeat(1, n_level, 1, 1) > lower) & (x.repeat(1, n_level, 1, 1) < upper)).float()
        disc_x = (1 - invalid_mask) * disc_x + invalid_mask * -1.0
        return disc_x
