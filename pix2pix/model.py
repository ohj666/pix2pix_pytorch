import torch.nn as nn
import torch
import torch.nn.functional as F


def down_sample(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False, apply_norm=True):
    layers = []
    layers.append(
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias))
    if apply_norm:
        layers.append(nn.BatchNorm2d(out_channels))
    layers.append(nn.LeakyReLU(0.2, inplace=True))
    return nn.Sequential(*layers)


def up_sample(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False, apply_dropout=True):
    layers = []
    layers.append(nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias))
    layers.append(nn.BatchNorm2d(out_channels))
    if apply_dropout:
        layers.append(nn.Dropout2d(p=0.5))
    layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.down_stack = nn.ModuleList([
            down_sample(3, 64, apply_norm=False), #output = (64,128,128)
            down_sample(64, 128),#output = (128,64,64)
            down_sample(128, 256),#ouput = (256,32,32)
            down_sample(256, 512),#output = (512,16,16)
            down_sample(512, 512),#output = (512,8,8)
            down_sample(512, 512),#output = (512,4,4)
            down_sample(512, 512),#output = (512,2,2)
            down_sample(512, 512, apply_norm=False),#output = (512,1,1)

        ])

        '''
        在这个生成器中，下采样和上采样的过程可以看做是对图像特征进行编码和解码的过程。在下采样过程中，
        通过对图像的卷积操作和池化操作，图像的尺寸逐渐减小，通道数逐渐增大，从而提取出越来越抽象的特征信息。
        在上采样过程中，通过反卷积操作和特征图的拼接，将编码得到的高层次特征还原成与原图相同尺寸的图像。
        在这个生成器中，第一次上采样的输入通道数是512，是因为它是与第一次下采样得到的512通道特征图进行拼接的。
        而在第二次上采样时，为了还原图像的细节信息，需要使用更多的特征信息进行重建，因此输入通道数增加到1024，这样可以让生成器更加准确地还原图像。
        类似地，之后每次上采样的输入通道数都是之前的两倍，以逐步减少特征图中的抽象信息，增加细节信息的还原。
        '''

        self.up_stack = nn.ModuleList([
            up_sample(512, 512, apply_dropout=True),  # (batch_size, 2, 2, 1024)
            up_sample(1024, 512, apply_dropout=True),  # (batch_size, 4, 4, 1024)
            up_sample(1024, 512, apply_dropout=True),  # (batch_size, 8, 8, 1024)
            up_sample(1024, 512),  # (batch_size, 16, 16, 1024)
            up_sample(1024, 256),  # (batch_size, 32, 32, 512)
            up_sample(512, 128),  # (batch_size, 64, 64, 256)
            up_sample(256, 192),  # (batch_size, 128, 128, 128)
        ])

        self.last = nn.ConvTranspose2d(256, 3, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        skips = []
        for down in self.down_stack:
            x = down(x)
            skips.append(x)

        skips = reversed(skips[:-1])

        for up, skip in zip(self.up_stack, skips):
            x = up(x)
            x = torch.cat([x, skip], dim=1)

        x = self.last(x)

        return x


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.down1 = down_sample(3 * 2, 64, apply_norm=False)
        self.down2 = down_sample(64, 128)
        self.down3 = down_sample(128, 256)
        self.zero_pad1 = nn.ZeroPad2d((1, 1, 1, 1))
        self.conv = nn.Conv2d(256, 512, kernel_size=4, stride=1, padding=0, bias=False)
        self.batch_norm1 = nn.BatchNorm2d(512)
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.zero_pad2 = nn.ZeroPad2d((1, 1, 1, 1))
        self.last = nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=0)

    def forward(self, inp, tar):
        x = torch.cat([inp, tar], dim=1)
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)

        x = self.zero_pad1(x)
        x = self.conv(x)
        x = self.batch_norm1(x)
        x = self.leaky_relu(x)
        x = self.zero_pad2(x)
        x = self.last(x)

        return x


class TestModel(nn.Module):
    def __init__(self):
        super(TestModel, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1024, 4, 2, 1, bias=False),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(1024, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),
            nn.ReLU(True),
        )

    def forward(self, input):
        return self.main(input)
