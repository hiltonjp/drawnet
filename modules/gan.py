import torch
from torch.optim import Adam
from torch.nn import Module


from tqdm import tqdm

from unet import UNet
from discriminators import Discriminator
from loss import GeneratorLoss

class GAN(Module):
    
    def __init__(self, generator, discriminator):
        super(GAN, self).__init__()
        self.__dict__.update(locals())


    def forward(self, x, mask):
        x_gen = self.generator(x)
        y, feats = self.discriminator(x_gen)

        return x_gen, y, feats





if __name__ == '__main__':
    gen = UNet()
    dis = Discriminator()

    gan = GAN(gen, dis)

    img1 = torch.randn(1, 3, 256, 256)
    img2 = torch.randn(1, 3, 256, 256)
    mask = torch.randn(1, 1, 256, 256)

    gen1 = gen(img1)
    # gen2, confidence2, feats2 = gan(img2)
    # comp = mask*img1 + (1-mask)*gen
    criterion = GeneratorLoss(dis)
    
    loss = criterion(gen1, img1, mask)
    print(loss)
