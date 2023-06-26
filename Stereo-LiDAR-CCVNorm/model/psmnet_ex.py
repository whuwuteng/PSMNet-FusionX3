"""
Stereo + LiDAR fusion: incorporate sparse disparity map into stereo matching network.
"""

import torch
import torch.nn as nn
from torch.autograd import Variable
import math

from .gcnet_conv import net_init, conv2d_bn, conv_res, conv3d_bn, deconv3d_bn, conv3d_ccvnorm, deconv3d_ccvnorm
from .gcnet_fun import myAdd3d

# PSMNet
from .submodule import feature_extraction, disparityregression

flag_bias_t = True
flag_bn = True
activefun_t = nn.ReLU(inplace=True)

# only make the PSMNet as a layer
class PSMNetEx(nn.Module):
    def __init__(self, maxdisp=192, num_F=32, channel=3):
        super(PSMNetEx, self).__init__()
        self.maxdisp = maxdisp
        self.F = num_F

        self.feature_extraction = feature_extraction(channel)

        self.dres01 = conv3d_bn(self.F*2, self.F, kernel_size=3, stride=1, flag_bias=flag_bias_t, bn=flag_bn, activefun=activefun_t)
        # add ReLU

        self.dres02 = conv3d_bn(self.F,   self.F, kernel_size=3, stride=1, flag_bias=flag_bias_t, bn=flag_bn, activefun=activefun_t)
        # add ReLU

        self.dres11 = conv3d_bn(self.F,   self.F, kernel_size=3, stride=1, flag_bias=flag_bias_t, bn=flag_bn, activefun=activefun_t)
        # add ReLU

        self.dres12 = conv3d_bn(self.F,   self.F, kernel_size=3, stride=1, flag_bias=flag_bias_t, bn=flag_bn, activefun=None)

        self.dres21 = conv3d_bn(self.F,   self.F * 2, kernel_size=3, stride=2, flag_bias=flag_bias_t, bn=flag_bn, activefun=activefun_t)
        # add ReLU

        self.dres22 = conv3d_bn(self.F * 2,   self.F * 2, kernel_size=3, stride=1, flag_bias=flag_bias_t, bn=flag_bn, activefun=None)

        self.dres23 = conv3d_bn(self.F * 2,   self.F * 2, kernel_size=3, stride=2, flag_bias=flag_bias_t, bn=flag_bn, activefun=activefun_t)
        # add ReLU

        self.dres24 = conv3d_bn(self.F * 2,   self.F * 2, kernel_size=3, stride=1, flag_bias=flag_bias_t, bn=flag_bn, activefun=activefun_t)
        # add ReLU

        self.dres25 = deconv3d_bn(self.F*2, self.F * 2, kernel_size=3, stride=2, flag_bias=flag_bias_t, bn=flag_bn, activefun=None)
        
        self.dres26 = deconv3d_bn(self.F*2, self.F, kernel_size=3, stride=2, flag_bias=flag_bias_t, bn=flag_bn, activefun=None)

        self.dres31 = conv3d_bn(self.F,   self.F * 2, kernel_size=3, stride=2, flag_bias=flag_bias_t, bn=flag_bn, activefun=activefun_t)
        # add ReLU

        self.dres32 = conv3d_bn(self.F * 2,   self.F * 2, kernel_size=3, stride=1, flag_bias=flag_bias_t, bn=flag_bn, activefun=None)

        self.dres33 = conv3d_bn(self.F * 2,   self.F * 2, kernel_size=3, stride=2, flag_bias=flag_bias_t, bn=flag_bn, activefun=activefun_t)
        # add ReLU

        self.dres34 = conv3d_bn(self.F * 2,   self.F * 2, kernel_size=3, stride=1, flag_bias=flag_bias_t, bn=flag_bn, activefun=activefun_t)
        # add ReLU

        self.dres35 = deconv3d_bn(self.F*2, self.F * 2, kernel_size=3, stride=2, flag_bias=flag_bias_t, bn=flag_bn, activefun=None)
        
        self.dres36 = deconv3d_bn(self.F*2, self.F, kernel_size=3, stride=2, flag_bias=flag_bias_t, bn=flag_bn, activefun=None)

        self.dres41 = conv3d_bn(self.F,   self.F * 2, kernel_size=3, stride=2, flag_bias=flag_bias_t, bn=flag_bn, activefun=activefun_t)
        # add ReLU

        self.dres42 = conv3d_bn(self.F * 2,   self.F * 2, kernel_size=3, stride=1, flag_bias=flag_bias_t, bn=flag_bn, activefun=None)

        self.dres43 = conv3d_bn(self.F * 2,   self.F * 2, kernel_size=3, stride=2, flag_bias=flag_bias_t, bn=flag_bn, activefun=activefun_t)
        # add ReLU

        self.dres44 = conv3d_bn(self.F * 2,   self.F * 2, kernel_size=3, stride=1, flag_bias=flag_bias_t, bn=flag_bn, activefun=activefun_t)
        # add ReLU

        self.dres45 = deconv3d_bn(self.F*2, self.F * 2, kernel_size=3, stride=2, flag_bias=flag_bias_t, bn=flag_bn, activefun=None)
        
        self.dres46 = deconv3d_bn(self.F*2, self.F, kernel_size=3, stride=2, flag_bias=flag_bias_t, bn=flag_bn, activefun=None)

        self.classif11 =  conv3d_bn(self.F,   self.F, kernel_size=3, stride=1, flag_bias=flag_bias_t, bn=flag_bn, activefun=activefun_t)
        # add ReLU

        self.classif12 =  conv3d_bn(self.F, 1, kernel_size=3, stride=1, flag_bias=flag_bias_t, bn=flag_bn, activefun=None)

        self.classif21 =  conv3d_bn(self.F,   self.F, kernel_size=3, stride=1, flag_bias=flag_bias_t, bn=flag_bn, activefun=activefun_t)
        # add ReLU

        self.classif22 =  conv3d_bn(self.F, 1, kernel_size=3, stride=1, flag_bias=flag_bias_t, bn=flag_bn, activefun=None)

        self.classif31 =  conv3d_bn(self.F,   self.F, kernel_size=3, stride=1, flag_bias=flag_bias_t, bn=flag_bn, activefun=activefun_t)
        # add ReLU

        self.classif32 =  conv3d_bn(self.F, 1, kernel_size=3, stride=1, flag_bias=flag_bias_t, bn=flag_bn, activefun=None)

        # initial 
        for m in self.modules():
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
                m.bias.data.zero_()


    def forward(self, left, right):

        refimg_fea     = self.feature_extraction(left)
        targetimg_fea  = self.feature_extraction(right)

        #matching
        cost = Variable(torch.FloatTensor(refimg_fea.size()[0], refimg_fea.size()[1]*2, self.maxdisp//4,  refimg_fea.size()[2],  refimg_fea.size()[3]).zero_()).cuda()

        #divide 4
        for i in range(self.maxdisp//4):
            if i > 0 :
             cost[:, :refimg_fea.size()[1], i, :,i:]   = refimg_fea[:,:,:,i:]
             cost[:, refimg_fea.size()[1]:, i, :,i:] = targetimg_fea[:,:,:,:-i]
            else:
             cost[:, :refimg_fea.size()[1], i, :,:]   = refimg_fea
             cost[:, refimg_fea.size()[1]:, i, :,:]   = targetimg_fea
        cost = cost.contiguous()

        # layer 1
        cost0 = self.dres01(cost)
        cost0 = self.dres02(cost0)

        # layer 2
        cost1 = self.dres11(cost0)
        cost1 = self.dres12(cost1)
        cost1 = cost1 + cost0

        # layer 3
        out1 = self.dres21(cost1)
        pre1 = self.dres22(out1)
        pre1 = nn.functional.relu(pre1, inplace=True)
        out1 = self.dres23(pre1)
        out1 = self.dres24(out1)
        out1 = self.dres25(out1) + pre1
        post1 = nn.functional.relu(out1, inplace=True)
        out1 = self.dres26(out1)
        out1 = out1 + cost1

        # layer 4
        out2 = self.dres31(out1)
        pre2 = self.dres32(out2)
        pre2 = nn.functional.relu(pre2 + post1, inplace=True)
        out2 = self.dres33(pre2)
        out2 = self.dres34(out2)
        out2 = self.dres35(out2) + pre1
        post2 = nn.functional.relu(out2, inplace=True)
        out2 = self.dres36(out2)
        out2 = out2 + cost1

        # layer 5
        out3 = self.dres41(out2)
        pre3 = self.dres42(out3)
        pre3 = nn.functional.relu(pre3 + post2, inplace=True)
        out3 = self.dres43(pre3)
        out3 = self.dres44(out3)
        out3 = self.dres45(out3) + pre2
        post3 = nn.functional.relu(out3, inplace=True)
        out3 = self.dres46(out3)
        out3 = out3 + cost1

        # last step 
        cost1 = self.classif11(out1)
        cost1 = self.classif12(out1)
        cost2 = self.classif21(out2)
        cost2 = self.classif22(cost2) + cost1
        cost3 = self.classif31(out3)
        cost3 = self.classif32(out3) + cost2

        if self.training:
            cost1 = nn.functional.upsample(cost1, [self.maxdisp,left.size()[2],left.size()[3]], mode='trilinear')
            cost2 = nn.functional.upsample(cost2, [self.maxdisp,left.size()[2],left.size()[3]], mode='trilinear')

            cost1 = torch.squeeze(cost1,1)
            pred1 = nn.functional.softmax(cost1,dim=1)
            pred1 = disparityregression(self.maxdisp)(pred1)

            cost2 = torch.squeeze(cost2,1)
            pred2 = nn.functional.softmax(cost2,dim=1)
            pred2 = disparityregression(self.maxdisp)(pred2)

        cost3 = nn.functional.upsample(cost3, [self.maxdisp,left.size()[2],left.size()[3]], mode='trilinear')
        cost3 = torch.squeeze(cost3,1)
        pred3 = nn.functional.softmax(cost3,dim=1)

        #For your information: This formulation 'softmax(c)' learned "similarity" 
        #while 'softmax(-c)' learned 'matching cost' as mentioned in the paper.
        #However, 'c' or '-c' do not affect the performance because feature-based cost volume provided flexibility.
        
        pred3 = disparityregression(self.maxdisp)(pred3)

        if self.training:
            return pred1, pred2, pred3
        else:
            return pred3