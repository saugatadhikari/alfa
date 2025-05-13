"""
Adapted from: https://github.com/javiribera/locating-objects-without-bboxes
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import torchvision

import eva_net_parts

from eva_net_parts import *

from elev_conv import *
from elev_deconv import *


class EvaNet(nn.Module):
    
    def __init__(self, batch_size, n_channels, n_classes, ultrasmall = False, device=torch.device('cuda')):
        
        """
        Instantiate a UNet network.
        :param n_channels: Number of input channels (e.g, 6 for two RGB)
        :param n_classes: Number of output classes
        :param device: Which torch device to use. Default: CUDA (GPU).
        """
        super(EvaNet, self).__init__()
        
        self.ultrasmall = ultrasmall
        self.device = device

        self.inc = inconv(n_channels, 1, 8)        
        self.down1 = down(8, 16)
        self.down2 = down(16, 32)
        
        if self.ultrasmall:
            self.down3 = down(32, 64, normaliz = False)
            self.up1 = up(64, 96, 32)
            self.up2 = up(32, 48, 16)
            self.up3 = up(16, 24, 8, activate = False)
        else:
            self.down3 = down(batch_size, 16, 16)
            self.down4 = down(batch_size, 16, 16)
            self.down5 = down(batch_size, 16, 32)
            self.down6 = down(batch_size, 32, 32)
            self.down7 = down(batch_size, 32, 32, normaliz = False)

            self.up1 = up(batch_size, 32, 32)
            self.up2 = up(batch_size, 32, 32)
            self.up3 = up(batch_size, 32, 16)
            self.up4 = up(batch_size, 16, 16)
            self.up5 = up(batch_size, 16, 16)
            self.up6 = up(batch_size, 16, 8)
            self.up7 = up(batch_size, 8, 8, activate = False)
        
        self.outc = outconv(8, 8, n_classes)
        self.out_nonlin = nn.Sigmoid()
#         self.out_nonlin = nn.Softmax()


    def forward(self, x, h):
        
        # print("Input: ", x.shape, h.shape)

        x1, h1 = self.inc(x, h)
        # print("In Conv: ", x1.shape, h1.shape)    
        x2, h2 = self.down1(x1, h1)
        # print("Down 1: ", x2.shape, h2.shape)
        x3, h3 = self.down2(x2, h2)
        # print("Down 2: ", x3.shape, h3.shape)
        x4, h4 = self.down3(x3, h3)
        # print("Down 3: ", x4.shape, h4.shape)

        
        if self.ultrasmall:
            x, h = self.up1(x4, x3, h4, h3)
            # print("Up 1: ", x.shape, h.shape)
            x, h = self.up2(x, x2, h, h2)
            # print("Up 2: ", x.shape, h.shape)
            x, h = self.up3(x, x1, h, h1)
            # print("Up 3: ", x.shape, h.shape)

        else:
            x5, h5 = self.down4(x4, h4)
            # print("Down 4: ", x4.shape, h4.shape)
            x6, h6 = self.down5(x5, h5)
            # print("Down 5: ", x5.shape, h5.shape)
            x7, h7 = self.down6(x6, h6)
            # print("Down 6:", x6.shape, h6.shape)
            x8, h8 = self.down7(x7, h7)
            # print("Down 7: ", x7.shape, h7.shape)
            
            x = self.up1(x8, x7, h8, h7)
            # print("Up 1: ", x.shape, h7.shape)
            x = self.up2(x, x6, h7, h6)
            # print("Up 2:", x.shape, h6.shape)
            x = self.up3(x, x5, h6, h5)
            # print("Up 3: ", x.shape, h5.shape)
            x = self.up4(x, x4, h5, h4)
            # print("Up 4: ", x.shape, h4.shape)
            x = self.up5(x, x3, h4, h3)
            # print("Up 5: ", x.shape, h3.shape)
            x = self.up6(x, x2, h3, h2)
            # print("Up 6: ", x.shape, h2.shape)
            x = self.up7(x, x1, h2, h)
            # print("Up 7: ", x.shape, h.shape)

        x_b = self.outc(x, h)
        # print("Out Conv: ", x.shape)
        x_a = self.out_nonlin(x_b)
        # print("Out Nonlin: ", x.shape)
        return x_b, x_a
