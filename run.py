import torch
from torch.functional import  F
from pix2pix.model import Generator,Discriminator, TestModel
from pix2pix.preprocessing import get_train_test_data, split_flip_crop_train_img, split_flip_crop_test_img, show_img
from torch import  optim, nn
BATCH_SIZE = 1
generator_lr = 0.001
discriminator_lr = 0.001
EPOCH = 10000

generator = Generator()
discriminator = Discriminator()
train, test = get_train_test_data()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print("---使用GPU进行训练---")
    generator.to(device)
    discriminator.to(device)

generator.train()
discriminator.train()
D_optim = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
G_optim = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
criterion = nn.BCELoss()
L1 = nn.L1Loss()
for i in range(EPOCH):
    for train_img,_ in test:




        input_img, target_img = split_flip_crop_train_img(train_img[0])
        # 生成器生产图片
        gen_output = generator(torch.unsqueeze(input_img, 0).to(device))
        # 判别器判别图片
        disc_generated_output = discriminator(torch.unsqueeze(input_img, 0).to(device), gen_output.to(device))

        # 计算判别器的损失值
        disc_real_output = discriminator(torch.unsqueeze(input_img, 0).to(device),
                                         torch.unsqueeze(target_img, 0).to(device))
        disc_real_loss = criterion(torch.ones_like(disc_real_output).to(device), disc_real_output)
        disc_gen_loss = criterion(torch.zeros_like(disc_generated_output), disc_generated_output)
        D_loss = disc_real_loss + disc_gen_loss
        D_loss.backward()
        D_optim.step()
        

        # 计算生成器损失值 判别器对生成出来的图片判别，再进行与1对比损失
        gen_gan_loss = criterion(torch.ones_like(disc_generated_output), disc_generated_output)
        G_out = gen_output.squeeze()
        gen_l1_loss = L1(G_out.to(device), target_img.to(device))

        G_loss = gen_l1_loss + gen_gan_loss
        G_loss.backward()
        G_optim.step()

        if i % 50 == 0:
            print('[%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f'
                  % (i, EPOCH,
                     D_loss.item(), G_loss.item()))




    break

