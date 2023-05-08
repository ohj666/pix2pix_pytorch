import random
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from matplotlib import pyplot as plt

IMAGE_SIZE = 256
TRAIN_ROOT = r"D:\over\new_email\new_email\traing_data\train"
TEST_ROOT = r"D:\over\new_email\new_email\traing_data\test"
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def get_train_test_data():
    train_data = torch.utils.data.DataLoader(datasets.ImageFolder(root=TRAIN_ROOT,
                                                                  transform=transforms.Compose([
                                                                      transforms.ToTensor(),
                                                                      transforms.Normalize((0.5, 0.5, 0.5),
                                                                                           (0.5, 0.5, 0.5)),
                                                                  ])
                                                                  ),
                                             batch_size=8,
                                             shuffle=True,
)

    test_data = torch.utils.data.DataLoader(datasets.ImageFolder(root=TEST_ROOT,
                                                            transform=transforms.Compose([
                                                              transforms.ToTensor(),
                                                              transforms.Normalize((0.5, 0.5, 0.5),
                                                                                   (0.5, 0.5, 0.5)),
                                                                  ])
                                                                  ),
                                       batch_size=1,
                                       shuffle=True)
    return train_data,test_data


def show_img(img):
    img = img * 0.5 + 0.5
    img = img.permute(1, 2, 0)
    plt.imshow(img)
    plt.show()


def split_flip_crop_train_img(image):
    real, rain_img = torch.chunk(image, 2, 3)
    transform = transforms.Compose([

        transforms.Resize(284),
        transforms.RandomCrop(256)
    ])
    flip = transforms.RandomHorizontalFlip(1)
    if random.random()>0.5:
        rain_img = flip(rain_img)
        real = flip(real)
    real = transform(real)
    rain_img = transform(rain_img)
    return real, rain_img


def split_flip_crop_test_img(image):
    real, rain_img = torch.chunk(image, 2, 3)
    transform = transforms.Compose([
        transforms.Resize(284),
        transforms.RandomCrop(256)
    ])
    real = transform(real)
    rain_img = transform(rain_img)
    return real, rain_img

