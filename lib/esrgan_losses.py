"""ESRGAN 손실. SRGAN 과 다른 점이 셋이다.

  1) 지각 손실을 **활성화 이전** VGG19 conv5_4 특징에서 잰다 (SRGAN 은 활성화 이후)
  2) 적대적 손실이 **상대적(RaGAN)** 이다 — 절대 진짜/가짜가 아니라
     "평균적인 가짜보다 얼마나 더 진짜 같은가" 를 본다
  3) 화소 손실이 L1 (SRGAN 은 MSE)

가중치는 논문 기본값: pixel 1e-2, perceptual 1.0, gan 5e-3
"""
import torch
import torch.nn as nn
from torchvision.models import vgg19

W = {'pixel': 1e-2, 'perceptual': 1.0, 'gan': 5e-3}
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class VGGFeature(nn.Module):
    """conv5_4 까지 (index 34). ReLU(35) 를 포함하지 않아 '활성화 이전' 이다."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(*list(vgg19(weights='DEFAULT').features)[:35]).eval()
        for p in self.net.parameters():
            p.requires_grad = False
        self.register_buffer('mean', MEAN)
        self.register_buffer('std', STD)

    def forward(self, x):
        return self.net((x - self.mean) / self.std)


class ESRGANLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.vgg = VGGFeature()
        self.l1 = nn.L1Loss()
        self.bce = nn.BCEWithLogitsLoss()

    def generator(self, d_real, d_fake, sr, hr):
        pixel = self.l1(sr, hr)
        perceptual = self.l1(self.vgg(sr), self.vgg(hr))
        ones, zeros = torch.ones_like(d_fake), torch.zeros_like(d_fake)
        # RaGAN: 생성자는 (가짜 - 평균진짜) 를 진짜로, (진짜 - 평균가짜) 를 가짜로 만들려 한다
        gan = 0.5 * (self.bce(d_real - d_fake.mean(), zeros)
                     + self.bce(d_fake - d_real.mean(), ones))
        total = W['pixel'] * pixel + W['perceptual'] * perceptual + W['gan'] * gan
        self.last = {'pixel': pixel.item(), 'perceptual': perceptual.item(), 'gan': gan.item(),
                     'w_pixel': W['pixel'] * pixel.item(),
                     'w_perceptual': W['perceptual'] * perceptual.item(),
                     'w_gan': W['gan'] * gan.item()}
        return total

    def discriminator(self, d_real, d_fake):
        ones, zeros = torch.ones_like(d_real), torch.zeros_like(d_real)
        return 0.5 * (self.bce(d_real - d_fake.mean(), ones)
                      + self.bce(d_fake - d_real.mean(), zeros))
