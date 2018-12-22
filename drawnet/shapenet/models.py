import torch.nn as nn
import matplotlib.pyplot as plt

import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=UserWarning)


class Generator(nn.Module):
    """Generator(im_size)

    A Generator designed to generate example images from an image distribution.

    Args:
        im_size (int): The desired size (width and height) of output images.
            This also set the size of the latent input vector.
    """

    def __init__(self, im_size=128):
        super(Generator, self).__init__()
        self.z_size = im_size
        # print(im_size)
        modules = []
        channels = im_size
        while channels > 1:

            modules.extend([
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv2d(in_channels=channels,
                          out_channels=channels,
                          kernel_size=5,
                          stride=1,
                          padding=2,
                          bias=True),
                nn.BatchNorm2d(channels),
                nn.PReLU(),
                nn.Conv2d(in_channels=channels,
                          out_channels=channels//2,
                          kernel_size=5,
                          stride=1,
                          padding=2,
                          bias=True),
                nn.BatchNorm2d(channels//2),
                nn.PReLU()
            ])
            channels //= 2
        modules.append(nn.Sigmoid())
        self.layers = nn.ModuleList(modules)
        # print(self.layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)

        return x


class Discriminator(nn.Module):
    """Discriminator(im_size)

    A Discriminator designed to determin whether an image is real of generated.

    Args:
        im_size (int): The size of the input image (width and height).
    """
    def __init__(self, im_size=128):
        super(Discriminator, self).__init__()

        modules = []
        channels = 1
        while im_size > 1:
            modules.extend([
                nn.Conv2d(in_channels=channels,
                          out_channels=channels,
                          kernel_size=5,
                          stride=1,
                          padding=2,
                          bias=True),
                nn.BatchNorm2d(channels),
                nn.LeakyReLU(0.2),
                nn.Conv2d(in_channels=channels,
                          out_channels=channels * 2,
                          kernel_size=2,
                          stride=2,
                          padding=0,
                          bias=True),
                nn.BatchNorm2d(channels * 2),
                nn.LeakyReLU(0.2)
            ])

            im_size //= 2
            channels *= 2

        modules.append(nn.Sigmoid())
        self.layers = nn.ModuleList(modules)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x