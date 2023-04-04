import torch
from pix2pix.model import Generator,Discriminator, TestModel
from utlis import loss
from pix2pix.preprocessing import get_train_test_data, split_flip_crop_train_img, split_flip_crop_test_img, show_img
BATCH_SIZE = 1
generator_lr = 0.001
discriminator_lr = 0.001
EPOCH = 10000

generator = Generator()
testmodel = TestModel()
# discriminator = Discriminator()
train, test = get_train_test_data()
dataiter = iter(train)
image, _ = next(dataiter)

# real, rain_img = split_flip_crop_train_img(image[0])
# show_img(real)
img_test = torch.randn(1, 3, 256, 256)
show_img(img_test[0])
gen_out = testmodel(img_test)
show_img(gen_out.detach()[0])
print(gen_out.size())




