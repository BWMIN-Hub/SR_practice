"""x2 EDSR 가중치를 만든다 — 14번(추론 실습) 페이지에서 scale 을 바꿔 보이려고.

x3 쌍(V2 의 g_LR)은 방사가 맞춰진 합성본이라 단순 bicubic 이 아니다. x2 쌍은
여기서 HR 을 bicubic 으로 절반 줄여 만든다. 14번 페이지는 화질이 아니라 타일링·
배치·정밀도 같은 추론 역학을 보이는 곳이라 이 차이는 문제가 되지 않는다.

    python tools/train_edsr_x2.py --gpu 0 --epochs 150
"""
import argparse
import glob
import os
import sys
import time

# torch.optim 이 torch.onnx 를 끌어오고, 그게 onnx -> ml_dtypes 로 이어지는데
# ml_dtypes 가 numpy 1.24 에서 AttributeError 를 낸다. torch 쪽 가드는 ImportError
# 만 잡으므로 통과해 버린다. onnx 를 막아 ImportError 로 바꿔 주면 가드가 작동한다.
sys.modules['onnx'] = None

import cv2
import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), 'lib'))
from sr_models import EDSR  # noqa: E402

ROOT = '/c/work/nst_work_dirs/G2601-SAR-Filtering/tmp_files/sr_data/V2'
SCALE, PATCH = 2, 96


def load(split):
    hrs, lrs = [], []
    for f in sorted(glob.glob(f'{ROOT}/{split}/HR/*.png')):
        hr = imageio.imread(f)
        h, w = hr.shape[0] // SCALE * SCALE, hr.shape[1] // SCALE * SCALE
        hr = hr[:h, :w]
        hrs.append(hr)
        lrs.append(cv2.resize(hr, (w // SCALE, h // SCALE),
                              interpolation=cv2.INTER_CUBIC))
    return np.stack(lrs), np.stack(hrs)


def batches(lr, hr, bs, rng):
    n = len(lr)
    for _ in range(n // bs):
        i = rng.integers(0, n, bs)
        x = rng.integers(0, lr.shape[1] - PATCH, bs)
        y = rng.integers(0, lr.shape[2] - PATCH, bs)
        a = np.stack([lr[i[k], x[k]:x[k] + PATCH, y[k]:y[k] + PATCH] for k in range(bs)])
        b = np.stack([hr[i[k], x[k] * SCALE:(x[k] + PATCH) * SCALE,
                         y[k] * SCALE:(y[k] + PATCH) * SCALE] for k in range(bs)])
        if rng.random() < 0.5:
            a, b = a[:, ::-1], b[:, ::-1]
        if rng.random() < 0.5:
            a, b = a[:, :, ::-1], b[:, :, ::-1]
        if rng.random() < 0.5:
            a, b = a.transpose(0, 2, 1, 3), b.transpose(0, 2, 1, 3)
        yield (torch.from_numpy(np.ascontiguousarray(a)).permute(0, 3, 1, 2).float(),
               torch.from_numpy(np.ascontiguousarray(b)).permute(0, 3, 1, 2).float())


@torch.no_grad()
def validate(net, lr, hr, dev, shave=SCALE):
    net.eval(); tot = 0.0
    for i in range(len(lr)):
        x = torch.from_numpy(lr[i]).permute(2, 0, 1).float().unsqueeze(0).to(dev)
        p = net(x)[0].clamp(0, 255).permute(1, 2, 0).cpu().numpy()
        g = hr[i].astype(np.float32)
        p, g = p[shave:-shave, shave:-shave], g[shave:-shave, shave:-shave]
        tot += 10 * np.log10(255.0 ** 2 / max(((p - g) ** 2).mean(), 1e-9))
    net.train()
    return tot / len(lr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpu', default='0')
    ap.add_argument('--epochs', type=int, default=150)
    ap.add_argument('--bs', type=int, default=16)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out = a.out or f'{here}/models/14_inference/checkpoints/edsr_x2.pt'
    os.makedirs(os.path.dirname(out), exist_ok=True)

    dev = f'cuda:{a.gpu}' if torch.cuda.is_available() else 'cpu'
    tr_lr, tr_hr = load('train')
    va_lr, va_hr = load('val')
    print(f'train {tr_lr.shape} -> {tr_hr.shape}   val {len(va_lr)}', flush=True)

    net = EDSR(scale=SCALE).to(dev).train()
    opt = torch.optim.Adam(net.parameters(), lr=a.lr)
    sch = torch.optim.lr_scheduler.MultiStepLR(
        opt, [int(a.epochs * 0.6), int(a.epochs * 0.85)], 0.5)
    loss_fn = nn.L1Loss()
    rng = np.random.default_rng(0)
    best, t0 = -1.0, time.time()
    for ep in range(1, a.epochs + 1):
        run = k = 0
        for x, y in batches(tr_lr, tr_hr, a.bs, rng):
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            l = loss_fn(net(x), y)
            l.backward(); opt.step()
            run += l.item(); k += 1
        sch.step()
        if ep % 5 == 0 or ep == a.epochs:
            p = validate(net, va_lr, va_hr, dev)
            tag = ''
            if p > best:
                best = p; torch.save(net.state_dict(), out); tag = '  <- saved'
            print(f'ep {ep:3d}  loss {run / k:6.3f}  val PSNR {p:6.3f}'
                  f'  {time.time() - t0:6.0f}s{tag}', flush=True)
    print(f'best val PSNR {best:.3f}  ->  {out}', flush=True)


if __name__ == '__main__':
    main()
