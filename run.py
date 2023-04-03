import torch
from pix2pix.model import Generator,Discriminator
from utlis import loss


BATCH_SIZE = 1
generator_lr = 0.001
discriminator_lr = 0.001
EPOCH = 10000

generator = Generator()
discriminator = Discriminator()


