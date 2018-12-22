import torch
import torch.nn as nn
import torch.nn.functional as f

class UNet(nn.Module):
    
    def __init__(self, size=16):
        super(UNet, self).__init__()

        self.top1 = nn.Conv2d(3, size, kernel_size=9, padding=4)
        self.top2 = nn.Conv2d(size, size, kernel_size=5, padding=2)

        self.down1 = Down(size, size*2, kernel_size=3, padding=1)
        self.down2 = Down(size*2, size*4, kernel_size=3, padding=1)
        self.down3 = Down(size*4, size*8, kernel_size=3, padding=1)
        self.down4 = Down(size*8, size*16, kernel_size=3, padding=1)

        self.up4 = Up(size*16, size*8, kernel_size=3, padding=1)
        self.up3 = Up(size*8, size*4, kernel_size=3, padding=1)
        self.up2 = Up(size*4, size*2, kernel_size=3, padding=1)
        self.up1 = Up(size*2, size, kernel_size=3, padding=1)

        self.out = nn.Conv2d(size, 3, kernel_size=9, padding=4)


    def forward(self, x):
        x1 = self.top1(x)
        x1 = f.relu(x1)
        x1 = self.top2(x1)
        x1 = f.relu(x1)

        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x4 = self.up4(x5, x4)
        x3 = self.up3(x4, x3)
        x2 = self.up2(x3, x2)
        x1 = self.up1(x2, x1)

        return self.out(x1)



class Down(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, padding=0, stride=1):
        super(Down, self).__init__()
        
        self.strided = nn.Conv2d(in_channels, 
                                 in_channels, 
                                 kernel_size=2, 
                                 padding=0, 
                                 stride=2)

        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, 
                               out_channels, 
                               kernel_size, 
                               padding=padding, 
                               stride=stride)

        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels,
                               out_channels,
                               kernel_size,
                               padding=padding,
                               stride=stride)
        
        self.bn3 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.strided(x)
        x = f.relu(x)
        x = self.bn1(x)

        x = self.conv1(x)
        x = f.relu(x)
        x = self.bn2(x)

        x = self.conv2(x)
        x = f.relu(x)
        
        return self.bn3(x)


class Up(nn.Module):
    
    def __init__(self, in_channels, out_channels, kernel_size, padding=0, stride=1):
        super(Up, self).__init__()

        self.resize_conv = nn.Conv2d(in_channels,
                                     out_channels,
                                     kernel_size=(3,3),
                                     padding=padding,
                                     stride=stride)

        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv1 = nn.Conv2d(in_channels,
                               out_channels,
                               kernel_size,
                               padding=padding,
                               stride=stride)

        self.bn2 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels,
                               out_channels,
                               kernel_size,
                               padding=padding,
                               stride=stride)

        self.bn3 = nn.BatchNorm2d(out_channels)


    def forward(self, x, cat):
        x = f.interpolate(x, scale_factor=2)
        x = self.resize_conv(x)
        x = f.relu(x)
        x = self.bn1(x)

        x = self.conv1(torch.cat((cat, x), dim=1))
        x = f.relu(x)
        x = self.bn2(x)

        x = self.conv2(x)
        x = f.relu(x)
        return self.bn3(x)
