"""Temporal MISR x3 — the target LR image plus a second input: LR images of other dates.

    from tsa_models import load_misr, run_tiled
    net = load_misr('misr_x3.pt')
    sr = run_tiled(net, target, second)   # (h, w, 3) uint8, (T, h, w, 3) uint8 -> (3h, 3w, 3) uint8

SwinIR layout (window attention, shifted windows every other block, residual groups).
Every attention is target -> second input: queries come from the target window (ws x ws
tokens), keys and values from the same window in every second-input image.  There is no
attention among target tokens and none among second-input tokens.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['TemporalMISR', 'load_misr', 'run_tiled']


def _win(x, ws):
    """(B, H, W, C) -> (B * nW, ws * ws, C)"""
    B, H, W, C = x.shape
    x = x.view(B, H // ws, ws, W // ws, ws, C).permute(0, 1, 3, 2, 4, 5)
    return x.reshape(-1, ws * ws, C)


def _unwin(x, ws, B, H, W):
    C = x.shape[-1]
    x = x.view(B, H // ws, W // ws, ws, ws, C).permute(0, 1, 3, 2, 4, 5)
    return x.reshape(B, H, W, C)


class CrossWindowAttention(nn.Module):
    """Target window -> the second-input windows at the same place, all dates."""

    def __init__(self, dim, heads, ws):
        super().__init__()
        self.h, self.ws = heads, ws
        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, dim * 2)
        self.proj = nn.Linear(dim, dim)
        # spatial relative position bias, shared by every date
        self.bias = nn.Parameter(torch.zeros((2 * ws - 1) ** 2, heads))
        nn.init.trunc_normal_(self.bias, std=.02)
        c = torch.stack(torch.meshgrid(torch.arange(ws), torch.arange(ws), indexing='ij')).flatten(1)
        r = (c[:, :, None] - c[:, None, :]).permute(1, 2, 0) + ws - 1
        self.register_buffer('rel', r[..., 0] * (2 * ws - 1) + r[..., 1], persistent=False)

    def forward(self, t, a, T, B, mask=None):
        """t: (B*nW, N, C)  a: (B*nW, T*N, C)  mask: (nW, N, N) or None"""
        Bw, N, C = t.shape
        nW, d = Bw // B, C // self.h
        # windows folded into the head axis, so the bias is not copied per batch
        q = self.q(t).view(B, nW, N, self.h, d).permute(0, 1, 3, 2, 4).reshape(B, nW * self.h, N, d)
        k, v = self.kv(a).view(B, nW, T * N, 2, self.h, d).permute(3, 0, 1, 4, 2, 5).reshape(
            2, B, nW * self.h, T * N, d)
        b = self.bias[self.rel].permute(2, 0, 1)[None]              # (1, h, N, N)
        if mask is not None:
            b = b + mask[:, None]                                    # (nW, h, N, N)
        b = b.expand(nW, -1, -1, -1).reshape(nW * self.h, N, N).repeat(1, 1, T)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=b.to(q.dtype))
        y = y.view(B, nW, self.h, N, d).permute(0, 1, 3, 2, 4).reshape(Bw, N, C)
        return self.proj(y)


class Block(nn.Module):
    def __init__(self, dim, heads, ws, shift, mlp=2.0):
        super().__init__()
        self.ws, self.shift = ws, shift
        self.nt = nn.LayerNorm(dim)
        self.attn = CrossWindowAttention(dim, heads, ws)
        self.n2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, int(dim * mlp)), nn.GELU(),
                                 nn.Linear(int(dim * mlp), dim))

    def _mask(self, H, W, dev):
        ws, s = self.ws, self.shift
        img = torch.zeros(1, H, W, 1, device=dev)
        n = 0
        for hs in (slice(0, -ws), slice(-ws, -s), slice(-s, None)):
            for wsl in (slice(0, -ws), slice(-ws, -s), slice(-s, None)):
                img[:, hs, wsl] = n; n += 1
        m = _win(img, ws).squeeze(-1)
        m = m[:, :, None] - m[:, None, :]
        return m.masked_fill(m != 0, -100.0).masked_fill(m == 0, 0.0)

    def forward(self, t, aw, T):
        """t: (B, H, W, C)  aw: second-input window tokens (unshifted, shifted), each (B*nW, T*N, C)"""
        B, H, W, C = t.shape
        x = self.nt(t)
        if self.shift:
            x = torch.roll(x, (-self.shift, -self.shift), (1, 2))
        xw = _win(x, self.ws)                                        # (B*nW, N, C)
        yw = aw[1 if self.shift else 0]
        mask = self._mask(H, W, t.device) if self.shift else None
        o = _unwin(self.attn(xw, yw, T, B, mask), self.ws, B, H, W)
        if self.shift:
            o = torch.roll(o, (self.shift, self.shift), (1, 2))
        t = t + o
        return t + self.mlp(self.n2(t))


class Group(nn.Module):
    """RSTB-style: several blocks + a conv, one residual around the group."""

    def __init__(self, dim, depth, heads, ws):
        super().__init__()
        self.blocks = nn.ModuleList([Block(dim, heads, ws, 0 if i % 2 == 0 else ws // 2)
                                     for i in range(depth)])
        self.conv = nn.Conv2d(dim, dim, 3, padding=1)

    def forward(self, t, aw, T):
        x = t
        for b in self.blocks:
            x = b(x, aw, T)
        return t + self.conv(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)


class SecondEncoder(nn.Module):
    """Each second-input image through the same convs; dates are not mixed."""

    def __init__(self, dim):
        super().__init__()
        self.c = nn.Sequential(nn.Conv2d(3, dim, 3, padding=1), nn.GELU(),
                               nn.Conv2d(dim, dim, 3, padding=1), nn.GELU(),
                               nn.Conv2d(dim, dim, 3, padding=1))

    def forward(self, a):
        B, T, C, H, W = a.shape
        return self.c(a.reshape(B * T, C, H, W)).view(B, T, -1, H, W)


class TemporalMISR(nn.Module):
    def __init__(self, dim=64, groups=4, depth=4, heads=4, ws=8, scale=3):
        super().__init__()
        self.ws, self.scale = ws, scale
        self.head = nn.Conv2d(3, dim, 3, padding=1)
        self.second = SecondEncoder(dim)
        self.anorm = nn.LayerNorm(dim)                # the second input does not change between blocks: normalise once
        self.groups = nn.ModuleList([Group(dim, depth, heads, ws) for _ in range(groups)])
        self.norm = nn.LayerNorm(dim)
        self.body_tail = nn.Conv2d(dim, dim, 3, padding=1)
        self.up = nn.Sequential(nn.Conv2d(dim, dim, 3, padding=1), nn.LeakyReLU(0.2, True),
                                nn.Conv2d(dim, 3 * scale ** 2, 3, padding=1),
                                nn.PixelShuffle(scale))

    def forward(self, target, second):
        B, _, h, w = target.shape
        p = (-h) % self.ws, (-w) % self.ws
        x, a = target, second
        if any(p):
            x = F.pad(x, (0, p[1], 0, p[0]), mode='reflect')
            a = F.pad(a.flatten(0, 1), (0, p[1], 0, p[0]), mode='reflect').view(
                B, second.shape[1], 3, h + p[0], w + p[1])
        f0 = self.head(x)
        t = f0.permute(0, 2, 3, 1)
        av = self.anorm(self.second(a).permute(0, 1, 3, 4, 2))          # (B, T, H, W, C)
        T, H, W, C = av.shape[1:]
        aw = []
        for s in (0, self.ws // 2):
            y = torch.roll(av, (-s, -s), (2, 3)) if s else av
            y = _win(y.reshape(B * T, H, W, C), self.ws)
            nW, N = y.shape[0] // (B * T), y.shape[1]
            aw.append(y.view(B, T, nW, N, C).permute(0, 2, 1, 3, 4).reshape(B * nW, T * N, C))
        for g in self.groups:
            t = g(t, aw, T)
        t = self.norm(t).permute(0, 3, 1, 2)
        out = self.up(self.body_tail(t) + f0)[:, :, :h * self.scale, :w * self.scale]
        return out + F.interpolate(target, scale_factor=self.scale, mode='bicubic',
                                   align_corners=False)


def load_misr(weight, device=None):
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    net = TemporalMISR()
    net.load_state_dict(torch.load(weight, map_location='cpu'))
    return net.to(device).eval()


def run_tiled(net, target, second, tile=128, ov=16):
    """target (h, w, 3) uint8, second (T, h, w, 3) uint8 -> (3h, 3w, 3) uint8.
    Tiles of `tile` px read `ov` px around themselves; the margins are dropped."""
    dev = next(net.parameters()).device
    h, w = target.shape[:2]
    t = torch.from_numpy(target.astype(np.float32) / 127.5 - 1).permute(2, 0, 1)
    x = torch.from_numpy(second.astype(np.float32) / 127.5 - 1).permute(0, 3, 1, 2)
    out = torch.zeros(3, h * 3, w * 3)
    step = tile - 2 * ov
    for y0 in range(0, h, step):
        for x0 in range(0, w, step):
            ya, xa = max(0, min(y0 - ov, h - tile)), max(0, min(x0 - ov, w - tile))
            yb, xb = min(h, ya + tile), min(w, xa + tile)
            with torch.no_grad():
                y = net(t[None, :, ya:yb, xa:xb].to(dev), x[None, :, :, ya:yb, xa:xb].to(dev))
            y = y.float().clamp(-1, 1)[0].cpu()
            ys, xs = y0 - ya, x0 - xa
            ye, xe = min(h, y0 + step) - ya, min(w, x0 + step) - xa
            out[:, y0 * 3:(ya + ye) * 3, x0 * 3:(xa + xe) * 3] = y[:, ys * 3:ye * 3, xs * 3:xe * 3]
    return ((out + 1) * 127.5).round().byte().permute(1, 2, 0).numpy()
