import torch
import torchvision.transforms as t

from torch.utils.data import DataLoader
from torch.optim import Adam, SGD

import cv2
import numpy as np

import os
import time
import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=UserWarning)

import dnnutil
from shapenet import *

def setup_data(args):
    transforms = t.Compose([
        t.Grayscale(),
        t.Resize((args.imsize, args.imsize)),
        t.RandomHorizontalFlip(),
        t.RandomVerticalFlip(),
        t.RandomAffine(degrees=45, shear=45),
        t.ToTensor()
    ])

    train_data = GANDataset(args.data, args.imsize, transform=transforms)
    test_data = GANDataset(args.data, args.imsize, transform=transforms)

    # kwargs = {'batch_size': args.batch_size, 'num_workers': 8}
    train_loader = DataLoader(train_data,
                              args.batch_size,
                              shuffle=True,
                              pin_memory=True)

    test_loader = DataLoader(test_data,
                             args.batch_size,
                             shuffle=True,
                             pin_memory=True)

    return train_loader, test_loader


def setup_generator(args):
    gen = Generator(args.imsize)

    if args.model != '':
        gen.load_state_dict(torch.load(args.model))

    return gen


def setup_discriminator(args):
    dis = Discriminator(args.imsize)

    if args.model != '':
        dis.load_state_dict(torch.load(args.model))

    return dis


def count_samples(args):
    if args.dst is None:
        return None

    index = 0
    for root, dirs, files in os.walk(args.dst):
        for _ in files:
            index += 1

    return index


def create_trainer(args, generator, discriminator):
    gen_trainer = GeneratorTrainer(generator,
                                   Adam(generator.parameters(), lr=args.lr, betas=(0.5, 0.999)),
                                   GeneratorLoss(discriminator),
                                   GeneratorAccuracy(discriminator))

    dis_trainer = DiscriminatorTrainer(discriminator,
                                       Adam(discriminator.parameters(), lr=args.lr,  betas=(0.5, 0.999)),
                                       DiscriminatorLoss(),
                                       DiscriminatorAccuracy())

    return GANTrainer(gen_trainer, dis_trainer)

def main(args):

    if args.dst is not None and not os.path.exists(args.dst):
        os.mkdir(args.dst)

    cur_sample = count_samples(args)

    # Create managers for each model.
    gen_dir = './runs/generator/'
    gen_manager = dnnutil.Manager(root=gen_dir, run_num=args.rid)
    gen_manager.set_description(args.note)

    dis_dir = './runs/discriminator/'
    dis_manager = dnnutil.Manager(root=dis_dir, run_num=args.rid)
    dis_manager.set_description(args.note)

    # Set up datasets
    train_loader, test_loader = setup_data(args)

    # Load up the models and create a Trainer object
    args = gen_manager.load_state(args, restore_lr=False)
    generator = setup_generator(args)
    # generator.cuda()

    args = dis_manager.load_state(args, restore_lr=False)
    discriminator = setup_discriminator(args)
    # discriminator.cuda()

    trainer = create_trainer(args, generator, discriminator)

    # Set up Training logs with console output.
    gen_log = dnnutil.TextLog(os.path.join(gen_manager.run_dir, 'gen_log.txt'),
                      console=True,
                      create_ok=True)
    dis_log = dnnutil.TextLog(os.path.join(dis_manager.run_dir, 'dis_log.txt'),
                      console=True,
                      create_ok=True)

    # Set up a checkpoint for each model
    gen_check = dnnutil.Checkpointer(gen_manager.run_dir)
    dis_check = dnnutil.Checkpointer(dis_manager.run_dir)

    for epoch in range(args.start, args.start + args.epochs):
        start = time.time()

        gtr_loss, dtr_loss, gtr_acc, dtr_acc = trainer.train(train_loader, epoch)
        # gtst_loss, dtst_loss, gtst_acc, dtst_acc = trainer.eval(test_loader, epoch)

        t = time.time() - start
        # gen_log.log(epoch, t, gtr_loss, gtr_acc, gtst_loss, gtst_acc)
        # dis_log.log(epoch, t, dtr_loss, dtr_acc, dtst_loss, dtst_acc)

        # gen_check.checkpoint(generator, gtst_loss, epoch)
        # dis_check.checkpoint(discriminator, dtst_loss, epoch)
        gen_manager.save_state(epoch, args.lr)
        dis_manager.save_state(epoch, args.lr)

        # generate a few samples
        generator.eval()
        images = generator(torch.randn(args.batch_size, args.imsize, 1, 1))
        images = images.squeeze(1)

        # make a few samples
        if args.dst is not None:
            for image in images.detach().cpu().numpy():
                print(np.min(image), np.max(image))
                image = (((image-np.min(image))*(255-0))/(np.max(image)-np.min(image)))
                print(np.min(image), np.max(image))
                # print(image)
                cv2.imwrite(os.path.join(args.dst, f'sample_{cur_sample}.png'),
                            image)
                cur_sample += 1
                break


if __name__ == '__main__':
    parser = dnnutil.basic_parser()

    parser.add_argument('--imsize', '-im',
                        type=int,
                        default=128,
                        help='The size of the image to generate. Default: 128.')

    parser.add_argument('data',
                        type=str,
                        help="The source folder of images.")

    parser.add_argument('--dst', '-o',
                        type=str,
                        help="The destination folder for sample images")

    args = parser.parse_args()

    main(args)