import torch
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset

import matplotlib.pyplot as plt

class RawImageDataset(Dataset):
    """A simple Dataset that stores raw images with user-defined transforms.

    Args:
        folder (str): The folder of images to use.
        transform (torchvision.transforms.Transform): A transformation to
            perform on the images in the folder, such as resizing, rotating,
            and flipping
    """

    def __init__(self, folder, transform):
        super(RawImageDataset, self).__init__()
        self.images = ImageFolder(folder, transform)

    def __getitem__(self, item):
        image, _ = self.images[item]
        return image

    def __len__(self):
        return len(self.images)


class GANDataset(Dataset):
    """A Dataset for training an Image Generative Adversarial Network.

    Args:
        z_size: The dimensionality of the z vector for the discriminator. This
            may be a single number or a tuple of dimensions.
        folder (str): The folder of images to use.
        transform (torchvision.transforms.Transform): A transformation to
            perform on the images in the folder, such as resizing, rotating,
            and flipping
    """

    def __init__(self, folder, z_size=128, transform=None):
        super(GANDataset, self).__init__()
        self.z_size = z_size
        self.images = ImageFolder(folder, transform)

    def __getitem__(self, item):
        if type(self.z_size) == int:
            batch1 = torch.randn(self.z_size, 1, 1)
            batch2 = torch.randn(self.z_size, 1, 1)
            fake = torch.randn(self.z_size, 1, 1)
        elif type(self.z_size) == tuple:
            batch1 = torch.randn(*self.z_size)
            batch2 = torch.randn(*self.z_size)
            fake = torch.randn(*self.z_size)
        else:
            raise ValueError("z_size must be either an int or tuple of ints.")
        images, _ = self.images[item]

        return images

    def __len__(self):
        return len(self.images)
