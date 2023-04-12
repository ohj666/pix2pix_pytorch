import torch
from torch.functional import  F
from pix2pix.model import Generator,Discriminator, TestModel
from pix2pix.preprocessing import get_train_test_data, split_flip_crop_train_img, split_flip_crop_test_img, show_img
from torch import  optim
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
for i in range(EPOCH):
    for train_img,_ in test:
        input_img, target_img = split_flip_crop_train_img(train_img[0])
        # 生成器生产图片
        gen_output = generator(torch.unsqueeze(input_img, 0).to(device))
        # 判别器判别图片
        disc_generated_output = discriminator(torch.unsqueeze(input_img, 0).to(device), gen_output.to(device))

        # 计算生成器损失值 判别器对生成出来的图片判别，再进行与1对比损失
        gen_gan_loss = F.binary_cross_entropy_with_logits(torch.ones_like(disc_generated_output), disc_generated_output)
        gen_l1_loss = F.l1_loss(gen_output, torch.unsqueeze(target_img, 0).to(device))
        G_loss = gen_l1_loss + gen_gan_loss
        G_loss.backward()
        G_optim.step()
        G_optim.zero_grad()
        # 计算判别器的损失值
        disc_real_loss = F.binary_cross_entropy_with_logits(torch.ones_like(torch.unsqueeze(target_img, 0)), torch.unsqueeze(target_img, 0))
        disc_gen_loss = F.binary_cross_entropy_with_logits(torch.zeros_like(disc_generated_output), disc_generated_output)
        disc_real_output = discriminator(torch.unsqueeze(input_img, 0).to(device), torch.unsqueeze(target_img, 0).to(device))
        D_loss = disc_real_loss + disc_gen_loss
        # D_loss.backward()
        # D_optim.step()

    break

