"""EDSR x2 for THEOS-2 — loading helper.

    from theos_models import load_theos_edsr
    net = load_theos_edsr('edsr_theos2_x2.pt')
    sr = net(x)          # x: (N, 3, H, W) in 0 - 1  ->  (N, 3, 2H, 2W) in 0 - 1

16 residual blocks of conv - LeakyReLU(0.2) - conv, 64 channels, no residual scaling,
then one conv to 256 channels and a pixel shuffle.  The network itself works in -1 - 1;
`forward` takes and returns 0 - 1 so it drops into the same tiling code as the others.
"""
import torch
import torch.nn as nn

__all__ = ['TheosEDSR', 'load_theos_edsr']


class _Block(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.c1 = nn.Conv2d(n, n, 3, padding=1)
        self.act = nn.LeakyReLU(0.2)
        self.c2 = nn.Conv2d(n, n, 3, padding=1)

    def forward(self, x):
        return x + self.c2(self.act(self.c1(x)))


class TheosEDSR(nn.Module):
    def __init__(self, n_feats=64, n_blocks=16, scale=2):
        super().__init__()
        self.head = nn.Conv2d(3, n_feats, 3, padding=1)
        self.body = nn.Sequential(*[_Block(n_feats) for _ in range(n_blocks)])
        self.body_tail = nn.Conv2d(n_feats, n_feats, 3, padding=1)
        self.up = nn.Conv2d(n_feats, n_feats * scale ** 2, 3, padding=1)
        self.shuffle = nn.PixelShuffle(scale)
        self.tail = nn.Conv2d(n_feats, 3, 3, padding=1)
        self.scale = scale

    def forward(self, x):
        x = x * 2 - 1
        h = self.head(x)
        h = h + self.body_tail(self.body(h))
        y = self.tail(self.shuffle(self.up(h)))
        return (y + 1) / 2


def load_theos_edsr(weight, device=None):
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    net = TheosEDSR().to(device)
    net.load_state_dict(torch.load(weight, map_location=device))
    return net.eval()
