import torch

from dnnutil.training import Trainer
from dnnutil.network import tocuda

class GeneratorTrainer(Trainer):
    """GeneratorTrainer(net, optim, loss_fn, accuracy_metric, epoch_size=None)

    Trainer for training a network to do data generation.

    Args:
        net (torch.nn.Module): An instance of a network that inherits from
            torch.nn.Module
        optim (torch.optim.Optimizer): An instance of an optimizer that 
            inherits from torch.optim.Optimizer.
        loss_fn (callable): A callable that calculates and returns a loss
            value. The loss value should be a single-element Tensor. For a
            generator model, the loss usually utilizes a Discriminator model.
        accuracy_metric (callable):  A callable that calculates and returns
            an accuracy value between 0 and 1.
        epoch_size (int): An optional epoch size, denoting the number of batches
            per epoch. If None, an epoch will consist of as many batches as 
            can be made from the dataset.
    """

    def train_batch(self, batch):
        """Trainer the Trainer's network on a single training batch.

        Args:
            batch (iterable): A tuple of inputs necessary to train the generator.
                This may be anything from a single input, such as a Tensor of 
                samplings from a gaussian distribution to a set of input images 
                with conditional information.

        Returns:
            loss (float): The mean loss over the batch.
            accuracy (float): the mean accuracy over the batch, between 0 and 1.
        """
        self.optim.zero_grad()

        inputs = tocuda(batch)
        generated = self.net(inputs)

        loss = self.loss_fn(generated)
        loss.backward()
        self.optim.step()

        loss = loss.item()
        with torch.no_grad():
            accuracy = self.measure_accuracy(generated)

        return loss, accuracy

    def test_batch(self, batch):
        """Evaluate the Trainer's network on a single training batch.
        
        Args:
            batch (iterable): A tuple of inputs necessary to evaluate the 
                generator. This may be anything from a single input, such 
                as a Tensor of samplings from a gaussian distribution to a 
                set of input images with conditional information.

        Returns:
            loss (float): The mean loss over the batch.
            accuracy (float): the mean accuracy over the batch, between 0 and 1.
        """
        with torch.no_grad():
            inputs = tocuda(batch)

            generated = self.net(inputs)
            loss = self.loss_fn(generated).item()
            accuracy = self.measure_accuracy(generated)

        return loss, accuracy


class DiscriminatorTrainer(Trainer):
    """DiscriminatorTrainer(net, optim, loss_fn, accuracy_metric, epoch_size=None)

    Trainer for training a network to do data generation.

    Args:
        net (torch.nn.Module): An instance of a network that inherits from
            torch.nn.Module
        optim (torch.optim.Optimizer): An instance of an optimizer that 
            inherits from torch.optim.Optimizer.
        loss_fn (callable): A callable that calculates and returns a loss
            value. The loss value should be a single-element Tensor.
        accuracy_metric (callable):  A callable that calculates and returns
            an accuracy value between 0 and 1.
        epoch_size (int): An optional epoch size, denoting the number of batches
            per epoch. If None, an epoch will consist of as many batches as 
            can be made from the dataset.
    """

    def train_batch(self, batch):
        """Trainer the Trainer's network on a single training batch.

        Args:
            batch (iterable): A tuple of inputs necessary to train the 
            discriminator.Often, this is a batch of images.

        Returns:
            loss (float): The mean loss over the batch.
            accuracy (float): the mean accuracy over the batch, between 0 and 1.
        """
        self.optim.zero_grad()
        batch, conf_true = batch
        inputs = tocuda(batch)

        confidence = self.net(inputs)

        loss = self.loss_fn(confidence, conf_true)
        loss.backward()
        self.optim.step()

        loss = loss.item()
        with torch.no_grad():
            accuracy = self.measure_accuracy(confidence, conf_true)

        return loss, accuracy

    def test_batch(self, batch):
        """Evaluate the Trainer's network on a single training batch.
        
        Args:
            batch (iterable): A tuple of inputs necessary to train the 
            discriminator. Often, this is a batch of images.

        Returns:
            loss (float): The mean loss over the batch.
            accuracy (float): the mean accuracy over the batch, between 0 and 1.
        """
        with torch.no_grad():
            batch, real = batch
            images = tocuda(batch)

            confidence = self.net(images)
            loss = self.loss_fn(confidence, real).item()
            accuracy = self.measure_accuracy(confidence, real)

        return loss, accuracy


class GANTrainer(Trainer):
    """GANTrainer(gen_trainer, dis_trainer)

    A wrapper trainer for training an adversarial system. Define the
    GeneratorTrainer and DiscriminatorTrainer separately.

    Args:
        gen_trainer (GeneratorTrainer): The trainer for the generative system.
        dis_trainer (DiscriminatorTrainer): The trainer for the discriminative
            system.
    """
    def __init__(self, gen_trainer, dis_trainer, epoch_size=None):
        super(GANTrainer, self).__init__(None, None, None, None)
        self.gen_trainer = gen_trainer
        self.dis_trainer = dis_trainer
        self.epoch_size = epoch_size

    def train(self, dataloader, epoch):
        self.gen_trainer.net.train()
        self.dis_trainer.net.train()
        return self._run_epoch(dataloader, epoch, training=True)

    def eval(self, dataloader, epoch):
        self.gen_trainer.net.eval()
        self.dis_trainer.net.eval()
        return self._run_epoch(dataloader, epoch, training=False)

    def train_batch(self, batch):
        """Train a Generative Adversarial Network on a single batch.
        Args:
            batch (iterator): A 4-tuple of Tensors. The first two tensors are
                the batches used to train the generator (two batches are needed
                to ensure that the generator is trained the same amount as the
                discriminator). The third tensor is the set of generator inputs
                to train the discriminator with false inputs. The last is a
                tensor of data from the real distribution (A set of real images,
                for instance) to train the discriminator on real inputs.

        Returns:
            gen_loss (float): The generator loss.
            dis_loss (float): The discriminator loss.
            gen_acc (float): The generator accuracy.
            dis_acc (float): The discriminator accuracy.
        """
        gen_batch1, gen_batch2, dis_batch_fake, dis_batch_real = batch

        # Train generator twice. This counteracts the fact that the
        # discriminator needs to be trained on two sets of images
        # (real and fake).
        gen_loss1, gen_acc1 = self.gen_trainer.train_batch(gen_batch1)
        gen_loss2, gen_acc2 = self.gen_trainer.train_batch(gen_batch2)
        gen_loss = (gen_loss1 + gen_loss2)/2
        gen_acc = (gen_acc1 + gen_acc2)/2

        # Generate false images using the generator
        gen = self.gen_trainer.net
        gen.eval()
        dis_batch_fake = gen(dis_batch_fake)
        gen.train()

        # Train discriminator with both real and false images.
        dis_loss1, dis_acc1 = self.dis_trainer.train_batch((dis_batch_fake, False))
        dis_loss2, dis_acc2 = self.dis_trainer.train_batch((dis_batch_real, True))
        dis_loss = (dis_loss1 + dis_loss2)/2
        dis_acc = (dis_acc1 + dis_acc2)/2

        return gen_loss, dis_loss, gen_acc, dis_acc

    def test_batch(self, batch):
        """Evaluate a Generative Adversarial Network on a single batch.
        Args:
            batch (iterator): A 4-tuple of Tensors. The first two tensors are
                the batches used to test the generator (two batches are needed
                to ensure that the generator is tested the same amount as the
                discriminator). The third tensor is the set of generator inputs
                to test the discriminator with false inputs. The last is a
                tensor of data from the real distribution (A set of real images,
                for instance) to test the discriminator on real inputs.

        Returns:
            gen_loss (float): The generator loss.
            dis_loss (float): The discriminator loss.
            gen_acc (float): The generator accuracy.
            dis_acc (float): The discriminator accuracy.
        """
        gen_batch1, gen_batch2, dis_batch_fake, dis_batch_real = batch

        # Test generator twice. This counteracts the fact that the
        # discriminator needs to be tested on two sets of images
        # (real and fake).
        gen_loss1, gen_acc1 = self.gen_trainer.test_batch(gen_batch1)
        gen_loss2, gen_acc2 = self.gen_trainer.test_batch(gen_batch2)
        gen_loss = (gen_loss1 + gen_loss2) / 2
        gen_acc = (gen_acc1 + gen_acc2) / 2

        # Generate false images using the generator
        gen = self.gen_trainer.net
        gen.eval()
        dis_batch_fake = gen(dis_batch_fake)
        gen.train()

        # Train discriminator with both real and false images.
        dis_loss1, dis_acc1 = self.dis_trainer.test_batch((dis_batch_fake, False))
        dis_loss2, dis_acc2 = self.dis_trainer.test_batch((dis_batch_real, True))
        dis_loss = (dis_loss1 + dis_loss2) / 2
        dis_acc = (dis_acc1 + dis_acc2) / 2

        return gen_loss, dis_loss, gen_acc, dis_acc

