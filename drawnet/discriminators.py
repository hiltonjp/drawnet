import torch 
from torch.nn import Module, Conv2d, BatchNorm2d, Dropout
import torch.nn.functional as f

class Discriminator(Module):
    
    def __init__(self, size=16):
        super(Discriminator, self).__init__()

        self.blocks = [ConvBlock(3, size)]
        self.downs = []
        for i in range(4):
            num_convs = 2 if i < 2 else 4
            self.downs.append(Conv2d(size*2**i, size*2**i, kernel_size=2, stride=2, padding=0))
            self.blocks.append(ConvBlock(size*2**i, size*2**(i+1)))

    def forward(self, x):
        feats = []
        for i in range(4):
            x = self.blocks[i](x)
            feats.append(x.clone())
            x = self.downs[i](x)

        return x, feats
        
        
class ConvBlock(Module):

    def __init__(self, 
                 input_channels, 
                 output_channels,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 dilation=1,
                 num_convs=2, 
                 bn=True, 
                 dropout=0):

        super(ConvBlock, self).__init__()

        self.convs = []
        for _ in range(num_convs):
            self.convs.append(
                Conv2d(input_channels, output_channels, kernel_size, stride, padding, dilation)
            )
            input_channels = output_channels

        self.bn = BatchNorm2d(output_channels, momentum=0.8) if bn else None
        self.drop = Dropout2d(p=dropout) if dropout else None

    def forward(self, x):
        for conv in self.convs:
            x = f.relu(conv(x))

        x = self.bn(x) if self.bn else x
        x = self.drop(x) if self.drop else x

        return x
