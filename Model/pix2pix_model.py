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



