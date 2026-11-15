"""MISR x3 — several LR images of the same place in, one HR image out.

    from misr_models import load_misr
    net = load_misr('misr_x3_t9.pt')
    sr = net(x)          # x: (N, 3, T=9, h, w) in -1 - 1  ->  (N, 3, 3h, 3w) in -1 - 1

The T inputs are stacked on a third axis and go through a 3D UNet; the T axis is
collapsed at every scale before the decoder.  Frame 0 upscaled with bicubic is added
at the end as a global residual.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['MISRNet', 'load_misr']


class Res3d(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.b = nn.Sequential(
            nn.Conv3d(c, c, 3, padding=1), nn.GroupNorm(8, c), nn.SiLU(),
            nn.Conv3d(c, c, 3, padding=1), nn.GroupNorm(8, c))

    def forward(self, x):
        return F.silu(x + self.b(x))


class Down3d(nn.Module):
    """Halves the space, leaves T alone."""

    def __init__(self, cin, cout):
        super().__init__()
        self.c = nn.Sequential(
            nn.Conv3d(cin, cout, 3, stride=(1, 2, 2), padding=1),
            nn.GroupNorm(8, cout), nn.SiLU())
        self.r = Res3d(cout)

    def forward(self, x):
        return self.r(self.c(x))


class Fuse(nn.Module):
    """Collapses the T axis in one conv."""

    def __init__(self, c, t=5):
        super().__init__()
        self.c = nn.Conv3d(c, c, (t, 1, 1))

    def forward(self, x):
        return self.c(x).squeeze(2)


class Up2d(nn.Module):
    def __init__(self, cin, cskip, cout):
        super().__init__()
        self.up = nn.ConvTranspose2d(cin, cout, 4, stride=2, padding=1)
        self.c = nn.Sequential(
            nn.Conv2d(cout + cskip, cout, 3, padding=1), nn.GroupNorm(8, cout),
            nn.SiLU(), nn.Conv2d(cout, cout, 3, padding=1), nn.GroupNorm(8, cout),
            nn.SiLU())

    def forward(self, x, skip):
        return self.c(torch.cat([self.up(x), skip], dim=1))


class MISRNet(nn.Module):
    def __init__(self, ch=48, scale=3, t=5, cin=3, ref=0):
        super().__init__()
        self.scale, self.ref = scale, ref
        c1, c2, c3 = ch, ch * 2, ch * 4
        self.stem = nn.Sequential(nn.Conv3d(cin, c1, 3, padding=1),
                                  nn.GroupNorm(8, c1), nn.SiLU(), Res3d(c1))
        self.d1 = Down3d(c1, c2)
        self.d2 = Down3d(c2, c3)
        self.mid = nn.Sequential(Res3d(c3), Res3d(c3))
        self.f0, self.f1, self.f2 = Fuse(c1, t), Fuse(c2, t), Fuse(c3, t)
        self.u1 = Up2d(c3, c2, c2)
        self.u2 = Up2d(c2, c1, c1)
        self.tail = nn.Sequential(
            nn.Conv2d(c1, c1, 3, padding=1), nn.SiLU(),
            nn.Conv2d(c1, cin * scale ** 2, 3, padding=1), nn.PixelShuffle(scale))

    def forward(self, x):
        """x: (B, C, T, h, w) -> (B, C, h*scale, w*scale)"""
        s0 = self.stem(x)
        s1 = self.d1(s0)
        s2 = self.mid(self.d2(s1))
        h = self.u1(self.f2(s2), self.f1(s1))
        h = self.u2(h, self.f0(s0))
        out = self.tail(h)
        # global residual: frame 0 upscaled with bicubic
        base = F.interpolate(x[:, :, self.ref], scale_factor=self.scale,
                             mode='bicubic', align_corners=False)
        return out + base


def load_misr(weight, t=9, device=None):
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    net = MISRNet(t=t)
    net.load_state_dict({k: v.float() for k, v in torch.load(weight, map_location='cpu').items()})
    return net.to(device).eval()
