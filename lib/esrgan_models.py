"""ESRGAN Generator / Discriminator (추론·실습용).

업스트림 [xinntao/ESRGAN](https://github.com/xinntao/ESRGAN) 의 RRDBNet 과 계층 이름이
같다. 배율만 인자로 받게 했다 — 업스트림은 F.interpolate(x2) 를 두 번 해서 ×4 고정이다.
  x4 : upconv1(x2) -> upconv2(x2)   (업스트림과 동일)
  x3 : upconv1(x3) 한 번
"""
import functools

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualDenseBlock_5C(nn.Module):
    def __init__(self, nf=64, gc=32):
        super().__init__()
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        return self.conv5(torch.cat((x, x1, x2, x3, x4), 1)) * 0.2 + x


class RRDB(nn.Module):
    """Residual in Residual Dense Block. BatchNorm 이 없는 것이 SRGAN 과의 큰 차이다."""

    def __init__(self, nf, gc=32):
        super().__init__()
        self.RDB1, self.RDB2, self.RDB3 = (ResidualDenseBlock_5C(nf, gc) for _ in range(3))

    def forward(self, x):
        return self.RDB3(self.RDB2(self.RDB1(x))) * 0.2 + x


class RRDBNetX(nn.Module):
    def __init__(self, in_nc=3, out_nc=3, nf=64, nb=8, gc=32, scale=3):
        super().__init__()
        self.scale = scale
        blk = functools.partial(RRDB, nf=nf, gc=gc)
        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1)
        self.RRDB_trunk = nn.Sequential(*[blk() for _ in range(nb)])
        self.trunk_conv = nn.Conv2d(nf, nf, 3, 1, 1)
        self.upconv1 = nn.Conv2d(nf, nf, 3, 1, 1)
        self.upconv2 = nn.Conv2d(nf, nf, 3, 1, 1)
        self.HRconv = nn.Conv2d(nf, nf, 3, 1, 1)
        self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        fea = self.conv_first(x)
        fea = fea + self.trunk_conv(self.RRDB_trunk(fea))
        if self.scale == 4:
            fea = self.lrelu(self.upconv1(F.interpolate(fea, scale_factor=2, mode='nearest')))
            fea = self.lrelu(self.upconv2(F.interpolate(fea, scale_factor=2, mode='nearest')))
        else:
            fea = self.lrelu(self.upconv1(F.interpolate(fea, scale_factor=self.scale, mode='nearest')))
        return self.conv_last(self.lrelu(self.HRconv(fea)))


class Discriminator(nn.Module):
    """VGG 형 판별자. RaGAN 은 로짓을 쓰므로 sigmoid 를 적용하지 않는다."""

    def __init__(self, nf=64):
        super().__init__()
        def blk(i, o, s):
            return [nn.Conv2d(i, o, 3, s, 1, bias=False), nn.BatchNorm2d(o), nn.LeakyReLU(0.2, True)]
        self.net = nn.Sequential(
            nn.Conv2d(3, nf, 3, 1, 1), nn.LeakyReLU(0.2, True),
            *blk(nf, nf, 2), *blk(nf, nf * 2, 1), *blk(nf * 2, nf * 2, 2),
            *blk(nf * 2, nf * 4, 1), *blk(nf * 4, nf * 4, 2),
            *blk(nf * 4, nf * 8, 1), *blk(nf * 8, nf * 8, 2),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(nf * 8, 1024, 1), nn.LeakyReLU(0.2, True), nn.Conv2d(1024, 1, 1))

    def forward(self, x):
        return self.net(x).view(x.size(0))


def load_esrgan(weight, scale=3, nb=8, device=None):
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    net = RRDBNetX(nf=64, nb=nb, scale=scale).to(device)
    net.load_state_dict(torch.load(weight, map_location=device))
    return net.eval()
