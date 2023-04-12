import torch.nn as nn
import torch
import torch.nn.functional as F
"""
在CycleGAN论文中，作者提到了两个损失函数：生成器损失和判别器损失。
其中，生成器损失由两个部分组成：对抗损失和重构损失。
LAMBDA=100是重构损失的权重系数，它用于控制生成器网络在生成图像时对输入图像的重构程度。
具体来说，对于一对输入图像 $A$ 和 $B$，生成器网络 $G$ 会将图像 $A$ 转换成类似于图像 $B$ 的图像 $A'$，同时将图像 $A'$ 转换回来
生成类似于图像 $A$ 的图像 $A''$。重构损失衡量了 $A$ 和 $A''$ 之间的相似度，即生成器网络是否能够保留输入图像的信息。
LAMBDA=100的值是通过实验得到的，可以控制重构损失的权重，以平衡对抗损失和重构损失的影响。

因此，LAMBDA=100的作用是平衡对抗损失和重构损失，从而实现更好的图像转换效果。
"""
def generator_loss(disc_generated_output, gen_output, target, LAMBDA=100):
    gan_loss = F.binary_cross_entropy_with_logits(disc_generated_output, torch.ones_like(disc_generated_output))
    l1_loss = F.l1_loss(gen_output, target)
    total_gen_loss = gan_loss.item() + (l1_loss.item() * LAMBDA)
    return total_gen_loss , gan_loss, l1_loss
def discriminator_loss(disc_real_output, disc_gen):
    real_loss = F.binary_cross_entropy_with_logits(disc_real_output, torch.ones_like(disc_real_output))
    generated_loss = F.binary_cross_entropy_with_logits(disc_gen, torch.zeros_like(disc_real_output))
    total_disc_loss = real_loss + generated_loss
    return total_disc_loss
def train_generator(generator, discriminator, generator_optimizer, input_image, target):
    with torch.no_grad():
        gen_output = generator(input_image)
    disc_generated_output = discriminator(input_image, gen_output)
    gen_loss, gan_loss, l1_loss = generator_loss(disc_generated_output, gen_output, target)
    generator_optimizer.zero_grad()
    gen_loss.backward()
    generator_optimizer.step()
    return gen_loss, gan_loss, l1_loss