import torch
import torch.nn as nn
import torch.nn.functional as f


class GeneratorLoss(nn.Module):
    """GeneratorLoss(discriminator)

    A loss that compares how well the generator fooled the discriminator.

    Args:
        discriminator (torch.nn.Module): A pytorch module that is the
            Discriminator the Generator is training against.
    """
    def __init__(self, discriminator):
        super(GeneratorLoss, self).__init__()
        self.__dict__.update(locals())
        self.loss = nn.BCELoss()

    def forward(self, generated):
        """Check the quality of the generated images.

        Args:
            generated (torch.FloatTensor): a tensor of generated data
                (usually images).

        Returns:
            loss (torch.FloatTensor): A one dimensional tensor that contains
            the loss representing how well the generator fooled
            the discriminator.
        """
        self.discriminator.eval()
        confidence = self.discriminator(generated)
        expected = torch.zeros_like(confidence)
        self.discriminator.train()

        loss = self.loss(confidence, expected)
        return loss


class DiscriminatorLoss(nn.Module):
    """DiscriminatorLoss()

    Convenience wrapper for binary crossentropy that may be either all ones or
    all zeros.

    """
    def __int__(self):
        super(DiscriminatorLoss, self).__init__()

    def forward(self, confidence, is_real):
        """Check how well the discriminator judges between real and fake data.

        Args:
            confidence (torch.FloatTensor): a tensor of floats between 0 and 1
                representing the confidence of the discriminator over a single
                batch of data.

            is_real (bool): a boolean stating whether the batch of data
                contained all real or all fake data.

        Returns:
            loss (torch.FloatTensor): A one dimensional tensor that is the loss,
            representing how well the discriminator can tell between real and
            false data.
        """
        expected = torch.zeros_like(confidence) if is_real \
                    else torch.ones_like(confidence)

        loss = f.binary_cross_entropy(confidence, expected)
        return loss
