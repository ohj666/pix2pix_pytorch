import torch
from torch.functional import  F
import torchvision
from pix2pix.model import Generator,Discriminator, TestModel
from pix2pix.preprocessing import get_train_test_data, split_flip_crop_train_img, split_flip_crop_test_img, show_img
from torch import  optim, nn
from tensorboardX import SummaryWriter
BATCH_SIZE = 1
generator_lr = 0.001
discriminator_lr = 0.001
EPOCH = 1000

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


def D_train(discriminator, generator, x, BCELoss, optimizer_D):
    input_img, target_img = split_flip_crop_train_img(x)
    input_img, target_img = input_img.to(device), target_img.to(device)
    D_real_r = discriminator(input_img, target_img)
    D_real_r = torch.sigmoid(D_real_r)
    D_real_loss = BCELoss(D_real_r, torch.ones(D_real_r.size()).to(device))
    g_output = generator(input_img)
    g_output = g_output.to(device)
    d_fake_f = discriminator(input_img, g_output)
    d_fake_f = torch.sigmoid(d_fake_f)
    D_fake_loss = BCELoss(d_fake_f, torch.zeros(d_fake_f.size()).to(device))
    D_loss = (D_real_loss + D_fake_loss) * 0.5
    D_loss.backward()
    optimizer_D.step()
    return D_loss.data.item()


def G_train(discriminator, generator, x, BCELoss, L1, optimizer_G, lamb=100):
    input_img, target_img = split_flip_crop_train_img(x)
    input_img, target_img = input_img.to(device), target_img.to(device)
    g_output = generator(input_img)
    g_output = g_output.to(device)
    d_fake_f = discriminator(input_img, g_output)
    d_fake_f = torch.sigmoid(d_fake_f)
    g_bce_loss = BCELoss(d_fake_f, torch.ones(d_fake_f.size()).to(device))
    g_l1_loss = L1(g_output, target_img)
    g_loss = g_l1_loss * 100 + g_bce_loss
    g_loss.backward()
    optimizer_G.step()
    return g_loss.data.item(), g_output


optimizer_G = optim.Adam(generator.parameters(), lr=generator_lr, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=discriminator_lr, betas=(0.5, 0.999))

for i in range(EPOCH):
    D_loss = []
    G_loss = []
    for epoch, (train_img, _) in enumerate(train):
        train_img = train_img.to(device)  # move the input tensor to the GPU device
        optimizer_D.zero_grad()
        optimizer_G.zero_grad()
        BCELoss = nn.BCELoss()
        L1 = nn.L1Loss()  # Pix2Pix论文中在传统GAN目标函数加上了L1
        d_loss = D_train(discriminator, generator, train_img, BCELoss, optimizer_D)
        g_loss, gan_out = G_train(discriminator, generator, train_img, BCELoss, L1, optimizer_G)
        D_loss.append(d_loss)
        G_loss.append(g_loss)
        print('[%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f'
              % (epoch, i,
                 d_loss, g_loss))
    if i % 10 == 0:
            print(f"完成第{i}次训练")
            writer = SummaryWriter("log_path/g_lr_0.1_d_lr_0.1_epoch_1000_2")
            writer.add_scalar('D_loss', sum(D_loss) / len(D_loss), i)
            writer.add_scalar('G_loss', sum(G_loss) / len(G_loss), i)
            X, _ = next(iter(test))
            src_grid = torchvision.utils.make_grid(X,  normalize=True)
            img,_ = split_flip_crop_test_img(X)
            gan_grid = torchvision.utils.make_grid(generator(img.to(device)))
            writer.add_image('src_img', src_grid, i)
            writer.add_image('gan_img', gan_grid, i)

torch.save(generator.state_dict(), 'generator_model.pth')


# for i in range(EPOCH):
#     for train_img,_ in test:
#         D_optim.zero_grad()
#         G_optim.zero_grad()
#         input_img, target_img = split_flip_crop_train_img(train_img[0])
#         # 生成器生产图片
#         gen_output = generator(torch.unsqueeze(input_img, 0).to(device))
#         # 判别器判别图片
#         disc_generated_output = discriminator(torch.unsqueeze(input_img, 0).to(device), gen_output.data)
#
#         # 计算判别器的损失值
#         disc_real_output = discriminator(torch.unsqueeze(input_img, 0).to(device),
#                                          torch.unsqueeze(target_img, 0).to(device))
#         disc_real_loss = criterion(torch.ones_like(disc_real_output).to(device), disc_real_output)
#         disc_gen_loss = criterion(torch.zeros_like(disc_generated_output).to(device), disc_generated_output)
#         D_loss = disc_real_loss + disc_gen_loss
#         # 计算生成器损失值 判别器对生成出来的图片判别，再进行与1对比损失
#         gen_gan_loss = criterion(torch.ones_like(disc_generated_output).to(device), disc_generated_output.data.to(device))
#         gen_l1_loss = L1(gen_output, torch.unsqueeze(target_img, 0).to(device))
#         G_loss = gen_l1_loss + gen_gan_loss
#         G_loss.backward()
#         D_loss.backward()
#         G_optim.step()
#         D_optim.step()
#
#         if i % 10 == 0:
#             print('[%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f'
#                   % (i, EPOCH,
#                      D_loss.item(), G_loss.item()))
#     print("="*20)





