import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
IMAGE_SIZE = 256
DATA_ROOT = r"D:\Googledownloads\traing_data\test"
train_data = torch.utils.data.DataLoader(datasets.ImageFolder(root=DATA_ROOT,
                                                              transform=transforms.Compose([
                                                                  transforms.Resize((286, 572)),
                                                                  transforms.CenterCrop((256, 512)),
                                                                  transforms.ToTensor(),
                                                                  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                                                  transforms.RandomHorizontalFlip()

                                                              ])
                                                              ),
                                         batch_size=1,
                                         shuffle=True)
dataiter = iter(train_data)
images, labels = dataiter.next()
img = images[0]
img = img / 2 + 0.5  # 反归一化
img = img.numpy().transpose((1, 2, 0))
plt.imshow(img)
plt.show()