import torch
import torch.nn as nn

def down_sample(in_channels,out_channels,kernel_size=4,stride=2,padding=1,bias=False,apply_norm=True):
    layers = []
    layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias))
    if apply_norm:
        layers.append(nn.BatchNorm2d(out_channels))
    layers.append(nn.LeakyReLU(0.2, inplace=True))
    return nn.Sequential(*layers)

def up_sample(in_channels,out_channels,kernel_size=4,stride=2,padding=1,bias=False,apply_dropout=True):
    layers = []
    layers.append(nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias))
    layers.append(nn.BatchNorm2d(out_channels))
    if apply_dropout:
        layers.append(nn.Dropout2d(p=0.5))
    layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.down_stack = nn.ModuleList([
            down_sample(64, 128, apply_norm=False),
            down_sample(128, 256),
            down_sample(256, 512),
            down_sample(512, 512),
            down_sample(512, 512),
            down_sample(512, 512),
            down_sample(512, 512),
            down_sample(512, 512),

        ])

        self.up_stack = nn.ModuleList([
            up_sample(512, 512, apply_dropout=True),
            up_sample(512, 512, apply_dropout=True),
            up_sample(512, 512, apply_dropout=True),
            up_sample(512, 256),
            up_sample(256, 128),
            up_sample(128, 64),
            up_sample(64, 4),
        ])

    def forward(self, x):
        skips = []
        for down in self.down_stack:
            x = down(x)
            skips.append(x)

        skips = reversed(skips[:-1])

        for up, skip in zip(self.up_stack, skips):
            x = up(x)
            x = torch.cat([x, skip], axis=1)

        x = self.last(x)
        return x

