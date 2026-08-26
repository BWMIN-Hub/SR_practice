"""SRGAN Generator (추론용). 학습 코드는 leftthomas/SRGAN 을 그대로 쓴다.

업스트림과 계층 이름·구조가 같아야 배포된 체크포인트가 그대로 실린다.
업스케일 블록만 손봤다 — 업스트림은 x2 블록을 log2(scale) 번 쌓아 x3 을 못 하므로,
2의 거듭제곱이 아니면 PixelShuffle(scale) 블록 하나를 쓴다.
"""
import math

import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(ch)
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(ch)

    def forward(self, x):
        return x + self.bn2(self.conv2(self.prelu(self.bn1(self.conv1(x)))))


class UpsampleBLock(nn.Module):
    def __init__(self, ch, up):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch * up ** 2, 3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(up)
        self.prelu = nn.PReLU()

    def forward(self, x):
        return self.prelu(self.pixel_shuffle(self.conv(x)))


class Generator(nn.Module):
    def __init__(self, scale_factor=3):
        super().__init__()
        self.block1 = nn.Sequential(nn.Conv2d(3, 64, 9, padding=4), nn.PReLU())
        for i in range(2, 8):
            setattr(self, f'block{i}', ResidualBlock(64))
        self.block7 = nn.Sequential(nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64))
        if scale_factor & (scale_factor - 1) == 0:
            up = [UpsampleBLock(64, 2) for _ in range(int(math.log(scale_factor, 2)))]
        else:
            up = [UpsampleBLock(64, scale_factor)]
        self.block8 = nn.Sequential(*up, nn.Conv2d(64, 3, 9, padding=4))

    def forward(self, x):
        b1 = self.block1(x)
        b = b1
        for i in range(2, 7):
            b = getattr(self, f'block{i}')(b)
        b7 = self.block7(b)
        return (torch.tanh(self.block8(b1 + b7)) + 1) / 2


def load_srgan(weight, scale=3, device=None):
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    net = Generator(scale).to(device)
    net.load_state_dict(torch.load(weight, map_location=device))
    return net.eval()


class Discriminator(nn.Module):
    """업스트림 그대로. 마지막에 전역 평균풀링을 거쳐 이미지당 값 하나를 낸다
    (PatchGAN 이 아니다)."""

    def __init__(self):
        super().__init__()
        def blk(i, o, stride=1, bn=True):
            m = [nn.Conv2d(i, o, 3, stride=stride, padding=1)]
            if bn:
                m.append(nn.BatchNorm2d(o))
            return m + [nn.LeakyReLU(0.2)]
        self.net = nn.Sequential(
            *blk(3, 64, bn=False), *blk(64, 64, 2),
            *blk(64, 128), *blk(128, 128, 2),
            *blk(128, 256), *blk(256, 256, 2),
            *blk(256, 512), *blk(512, 512, 2),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(512, 1024, 1), nn.LeakyReLU(0.2), nn.Conv2d(1024, 1, 1))

    def forward(self, x):
        return torch.sigmoid(self.net(x).view(x.size(0)))
