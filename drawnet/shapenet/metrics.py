import torch
import torch.nn as nn
import torch.nn.functional as f


class GeneratorAccuracy(nn.Module):
    """GeneratorAccuracy(discriminator)

    Measures how well the generator's images fool the discriminator.

    Args:
        discriminator (torch.nn.Module): The discriminator that the generator
            is training against.
    """

    def __init__(self, discriminator):
        super(GeneratorAccuracy, self).__init__()
        self.discriminator = discriminator

    def forward(self, images):
        """Measure the average discriminator score of the generated images.

        Args:
            images (torch.FloatTensor): A tensor of generated images, of size
                (batch, 3, h, w)

        Returns:
            accuracy (float): A number between 0 and 1, where 1 means the
                generator completely fooled the discriminator.

        """
        return torch.mean(self.discriminator(images)).item()


class DiscriminatorAccuracy(nn.Module):

    def forward(self, confidence, is_real):
        """Measures how accurate the discriminator is in its predictions.

        Args:
            confidence (torch.FloatTensor): a tensor of floats between 0 and 1
                representing the confidence of the discriminator over a single
                batch of data.

            is_real (bool): a boolean stating whether the batch of data
                contained all real or all fake data.

        Returns:
            accuracy (float): A float between 0 and 1, representing how well the
                discriminator can tell between real and false data, on average.
                Higher is better.
        """
        expected = torch.zeros_like(confidence) if is_real \
                    else torch.ones_like(confidence)

        return f.l1_loss(confidence, expected).item()
