import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms

IMAGE_SIZE = 256
DATA_ROOT =""
datasets.ImageFolder(root = DATA_ROOT,
                     transform=transforms.Compose([
                         transforms.Resize(286),
                         transforms.CenterCrop(256),
                         transforms.ToTensor(),
                         transforms.RandomHorizontalFlip(),

                     ]

                     ))