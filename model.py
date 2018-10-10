import torch
import torch.nn as nn
import torch.nn.functional as f
import random
import numpy as np

from torchvision import transforms, utils, datasets

import matplotlib.pyplot as plt

class DrawNet(nn.Module):

    def __init__(self):
        super(DrawNet, self).__init__()
        # x, _ = dataset[0]
        input = 3
        output = 3

        self.down1 = Down(input, 32, (7, 7), padding=(3, 3))    # 1     128
        self.down2 = Down(32, 32, (5, 5), padding=(2, 2))       # 2     64
        self.down3 = Down(32, 64, (5, 5), padding=(2, 2))       # 3     32
        self.down4 = Down(64, 64, (3, 3), padding=(1, 1))       # 4     16
        self.down5 = Down(64, 128, (3, 3), padding=(1, 1))      # 5     8
        self.down6 = Down(128, 128, (3, 3), padding=(1, 1))     # 6     4
        self.down7 = Down(128, 128, (3, 3), padding=(1, 1))     # 7     2
        self.down8 = Down(128, 128, (3, 3), padding=(1, 1))     # 8     1
        self.up1 = Up(128+128, 128, (3, 3), padding=(1, 1))     # 7     2
        self.up2 = Up(128+128, 128, (3, 3), padding=(1, 1))     # 6     4
        self.up3 = Up(128+128, 128, (3, 3), padding=(1, 1))     # 5     8
        self.up4 = Up(128+64, 64, (3, 3), padding=(1, 1))       # 4     16
        self.up5 = Up(64+64, 64, (3, 3), padding=(1, 1))       # 3     32
        self.up6 = Up(64+32, 32, (5, 5), padding=(2, 2))       # 2     64
        self.up7 = Up(32+32, 32, (5, 5), padding=(2, 2))        # 1     128
        self.up8 = Up(32+3, output, (7, 7), padding=(3, 3))     # 0     256


    def forward(self, *args):
        image, mask, motif = args

        # encode
        image1, mask1, motif1 = self.down1(image, mask, motif)
        image2, mask2, motif2 = self.down2(image1, mask1, motif1)
        image3, mask3, motif3 = self.down3(image2, mask2, motif2)
        image4, mask4, motif4 = self.down4(image3, mask3, motif3)
        image5, mask5, motif5 = self.down5(image4, mask4, motif4)
        image6, mask6, motif6 = self.down6(image5, mask5, motif5)
        image7, mask7, motif7 = self.down7(image6, mask6, motif6)
        image8, mask8, motif8 = self.down8(image7, mask7, motif7)

        # decode
        image9, mask9, motif9 = self.up1(image8, mask8, motif8, image7, mask7, motif7)
        image10, mask10, motif10 = self.up2(image9, mask9, motif9, image6, mask6, motif6)
        image11, mask11, motif11 = self.up3(image10, mask10, motif10, image5, mask5, motif5)
        image12, mask12, motif12 = self.up4(image11, mask11, motif11, image4, mask4, motif4)
        image13, mask13, motif13 = self.up5(image12, mask12, motif12, image3, mask3, motif3)
        image14, mask14, motif14 = self.up6(image13, mask13, motif13, image2, mask2, motif2)
        image15, mask15, motif15 = self.up7(image14, mask14, motif14, image1, mask1, motif1)
        image16, mask16, motif16 = self.up8(image15, mask15, motif15, image, mask, motif)

        return image16


class Down(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super(Down, self).__init__()
        self.pconv = PartialConv(in_channels, out_channels, kernel_size, stride=2, padding=padding)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=2, padding=padding)
        self.relu = nn.ReLU()
        self.norm = nn.BatchNorm2d(out_channels, momentum=0.8)

    def forward(self, *input):
        x, mask, motif = input
        x, mask = self.pconv(x+motif, mask)
        motif = self.conv(motif)
        x = self.relu(x)
        x = self.norm(x)
        return x, mask, motif


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super(Up, self).__init__()
        # self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)
        self.pconv = PartialConv(in_channels, out_channels, kernel_size, padding=padding)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.lrelu = nn.LeakyReLU(0.2)
        self.norm = nn.BatchNorm2d(out_channels)

    def forward(self, *input):
        x, mask, motif, x_skip, mask_skip, motif_skip = input

        #skip links
        x, mask, motif = f.interpolate(x, scale_factor=2), f.interpolate(mask, scale_factor=2), f.interpolate(motif, scale_factor=2) 
        print(x.size(), x_skip.size())
        x = torch.cat((x, x_skip), dim=1)
        mask = torch.cat((mask, mask_skip), dim=1)
        motif = torch.cat((motif, motif_skip), dim=1)
        
        x, mask = self.pconv(x, mask)
        motif = self.conv(motif)

        x = self.lrelu(x)
        x = self.norm(x)
        return x, mask, motif


class PartialConv(nn.Module):
    # reference:
    # Image Inpainting for Irregular Holes Using Partial Convolutions
    # http://masc.cs.gmu.edu/wiki/partialconv/show?time=2018-05-24+21%3A41%3A10
    # https://github.com/naoto0804/pytorch-inpainting-with-partial-conv/blob/master/net.py
    # https://github.com/SeitaroShinagawa/chainer-partial_convolution_image_inpainting/blob/master/common/net.py
    # mask is binary, 0 is holes; 1 is not
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=True):

        super(PartialConv, self).__init__()
        random.seed(0)
        self.feature_conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                                      padding, dilation, groups, bias)

        nn.init.kaiming_normal_(self.feature_conv.weight)

        self.mask_conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                                   padding, dilation, groups, bias=False)
        torch.nn.init.constant_(self.mask_conv.weight, 1.0)

        for param in self.mask_conv.parameters():
            param.requires_grad = False

    def forward(self, *args):
        x, mask = args
        output = self.feature_conv(x * mask)
        if self.feature_conv.bias is not None:
            output_bias = self.feature_conv.bias.view(1, -1, 1, 1).expand_as(output)
        else:
            output_bias = torch.zeros_like(output)

        with torch.no_grad():
            output_mask = self.mask_conv(mask)  # mask sums

        no_update_holes = output_mask == 0
        # because those values won't be used , assign a easy value to compute
        mask_sum = output_mask.masked_fill_(no_update_holes, 1.0)

        output_pre = (output - output_bias) / mask_sum + output_bias
        output = output_pre.masked_fill_(no_update_holes, 0.0)

        new_mask = torch.ones_like(output)
        new_mask = new_mask.masked_fill_(no_update_holes, 0.0)
        return output, new_mask


if __name__ == '__main__':
    x = torch.from_numpy(np.random.randn(10, 3, 256, 256)).float()
    mask = torch.from_numpy(np.random.randn(10, 1, 256, 256)).float()
    motif = torch.from_numpy(np.random.randn(10, 1, 256, 256)).float()
    mask = (mask > 3).float()
    mask = torch.cat((mask, mask, mask), 1)
    motif = torch.cat((motif, motif, motif), 1)

    # conv1 = PartialConv(3, 32, (3, 3), stride=2, padding=1)
    # conv2 = PartialConv(32, 32, (3, 3), stride=2, padding=1)
    # x, mask = conv1(x, mask)
    # print((torch.ones_like(mask) - mask).sum())
    # x, mask = conv2(x, mask)
    # print((torch.ones_like(mask) - mask).sum())
    # x, mask = conv2(x, mask)
    # print((torch.ones_like(mask) - mask).sum())
    # x, mask = conv2(x, mask)
    # print((torch.ones_like(mask) - mask).sum())
    # x, mask = conv2(x, mask)
    # print((torch.ones_like(mask) - mask).sum())

    net = DrawNet()
    print(np.sum([np.prod(p.size()) for p in net.parameters()]))
    y = net(x, mask, motif)
    plt.imshow(x)
    plt.imshow(y)
