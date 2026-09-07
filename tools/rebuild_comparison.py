"""비교 자산 전체(val*, test*)와 metrics.csv 를 현재 체크포인트로 다시 만든다.

val* : 실습 검증 패치. LR/Bicubic/HR/overview + 모델 7개
test*: 인천 두 구역. SR 6003px 중 가운데 768px 만 잘라 올린다 (전체는 너무 크다)

    python tools/rebuild_comparison.py
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np
import torch

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, f'{HERE}/lib')

import sr_utils as U                                            # noqa: E402
from esrgan_models import load_esrgan                            # noqa: E402
from hat_models import hat_upscale, load_hat                     # noqa: E402
from sr_models import load_edsr                                  # noqa: E402
from srcnn_models import load_srcnn, srcnn_upscale               # noqa: E402
from srgan_models import load_srgan                              # noqa: E402
from swinir_models import load_swinir                            # noqa: E402
from vdsr_models import load_vdsr, vdsr_upscale                  # noqa: E402

CK = f'{HERE}/models'
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
SCALE = 3
# 인천 두 구역 — 기존 자산과 같은 자리 (SR 6003px 기준 가운데 768)
TEST_SRC = (f'{HERE}/dataset/test/S2_2024-05-16_20240516T021611_'
            '20240516T022028_T52SBG_Incheon_RGB_8bit.tif')
TEST_WIN = {'test1': (516, 516), 'test2': (516, 516)}
TEST_TILE = {'test1': 'incheon_600.png', 'test2': 'incheon2_600.png'}


@torch.no_grad()
def t255(net, lr):
    t = torch.from_numpy(lr.transpose(2, 0, 1)).float()[None].to(DEV)
    return net(t).clamp(0, 255).round()[0].cpu().numpy().transpose(1, 2, 0).astype('uint8')


@torch.no_grad()
def t01(net, lr):
    t = torch.from_numpy(lr.transpose(2, 0, 1)).float()[None].to(DEV) / 255
    return (net(t).clamp(0, 1)[0].cpu().numpy().transpose(1, 2, 0) * 255).round().astype('uint8')


def build():
    out = []
    n = load_srcnn(f'{CK}/01_srcnn_x3/checkpoints/srcnn_x3.pth', device=DEV)
    out.append(('SRCNN', lambda lr, n=n: srcnn_upscale(n, lr), n))
    n = load_vdsr(f'{CK}/02_vdsr_x3/checkpoints/vdsr_x3.pth', device=DEV)
    out.append(('VDSR', lambda lr, n=n: vdsr_upscale(n, lr), n))
    n = load_edsr(f'{CK}/03_edsr_x3/checkpoints/edsr_x3.pt', device=DEV)
    out.append(('EDSR', lambda lr, n=n: t255(n, lr), n))
    n = load_srgan(f'{CK}/04_srgan_x3/checkpoints/srgan_g_x3.pth', device=DEV)
    out.append(('SRGAN', lambda lr, n=n: t01(n, lr), n))
    n = load_esrgan(f'{CK}/05_esrgan_x3/checkpoints/esrgan_g_x3.pth', device=DEV)
    out.append(('ESRGAN', lambda lr, n=n: t01(n, lr), n))
    n = load_swinir(f'{CK}/06_swinir_x3/checkpoints/swinir_x3.pth', device=DEV)
    out.append(('SwinIR', lambda lr, n=n: t01(n, lr), n))
    n = load_hat(f'{CK}/07_hat_x3/checkpoints/hat_x3.pth', device=DEV)
    out.append(('HAT', lambda lr, n=n: hat_upscale(n, lr), n))
    return out


def sr_tiled(fn, lr, tile=192, pad=24):
    h, w = lr.shape[:2]
    if max(h, w) <= tile:
        return fn(lr)
    out = np.zeros((h * SCALE, w * SCALE, 3), np.uint8)
    for y0 in range(0, h, tile):
        for x0 in range(0, w, tile):
            y1, x1 = min(y0 + tile, h), min(x0 + tile, w)
            py0, px0 = max(y0 - pad, 0), max(x0 - pad, 0)
            py1, px1 = min(y1 + pad, h), min(x1 + pad, w)
            sr = fn(lr[py0:py1, px0:px1])
            cy, cx = (y0 - py0) * SCALE, (x0 - px0) * SCALE
            out[y0 * SCALE:y1 * SCALE, x0 * SCALE:x1 * SCALE] = \
                sr[cy:cy + (y1 - y0) * SCALE, cx:cx + (x1 - x0) * SCALE]
    return out


def main():
    import imageio.v2 as io
    ap = argparse.ArgumentParser()
    ap.add_argument('--vals', nargs='+',
                    default=['val1=AOI_Paris_1_6_y0064_x0192',
                             'val2=AOI_Seoul_14_y0256_x0128',
                             'val3=AOI_Busan_7_y0064_x0192',
                             'val4=AOI_Barcelona_2_y0384_x0064'])
    a = ap.parse_args()

    res = f'{HERE}/results/comparison'
    os.makedirs(res, exist_ok=True)
    models = build()
    meta, rows = {}, []

    # ── 검증 패치 ────────────────────────────────────────────────
    d = f'{HERE}/dataset/validation'
    acc = {name: [] for name, _, _ in models}
    acc['Bicubic'] = []
    for spec in a.vals:
        key, stem = spec.split('=', 1)
        lr = io.imread(f'{d}/LR_bicubic/X{SCALE}/{stem}x{SCALE}.png')
        hr = io.imread(f'{d}/HR/{stem}.png')
        out = f'{res}/{key}'
        os.makedirs(out, exist_ok=True)
        io.imwrite(f'{out}/LR.png', U.nearest(lr, SCALE))
        io.imwrite(f'{out}/Bicubic.png', U.bicubic(lr, SCALE))
        io.imwrite(f'{out}/HR.png', hr)
        io.imwrite(f'{out}/overview.png', hr)
        for name, fn, _ in models:
            io.imwrite(f'{out}/{name}.png', fn(lr))
        h, w = hr.shape[:2]
        meta[key] = dict(title=f'validation {key[3:]} — {stem}', full=[h, w],
                         overview=[h, w], origin=[0, 0], extent=[w, h], has_hr=True)
        print(f'{key}: {stem}')

    # 지표는 검증 10장 전체로 낸다 (페이지 표와 같은 잣대)
    stems = sorted(f[:-4] for f in os.listdir(f'{d}/HR') if f.endswith('.png'))
    data = [(io.imread(f'{d}/LR_bicubic/X{SCALE}/{s}x{SCALE}.png'),
             io.imread(f'{d}/HR/{s}.png')) for s in stems]
    for name, fn in [('Bicubic', lambda lr: U.bicubic(lr, SCALE))] + \
                    [(n, f) for n, f, _ in models]:
        v = [U.score(fn(lr), hr) for lr, hr in data]
        p = float(np.mean([x[0] for x in v])); s = float(np.mean([x[1] for x in v]))
        nm = dict((n, m) for n, _, m in models).get(name)
        par = '' if nm is None else sum(x.numel() for x in nm.parameters()) / 1e6
        rows.append((name, p, s, par))
        print(f'  {name:8s} PSNR {p:.4f}  SSIM {s:.4f}')

    # ── 인천 두 구역 ──────────────────────────────────────────────
    import rasterio
    for key, (ox, oy) in TEST_WIN.items():
        lr_full = io.imread(f'{HERE}/dataset/test/{TEST_TILE[key]}')
        out = f'{res}/{key}'
        os.makedirs(out, exist_ok=True)
        big = U.bicubic(lr_full, SCALE)
        c = slice(ox, ox + 768)
        io.imwrite(f'{out}/LR.png', U.nearest(lr_full, SCALE)[c, c])
        io.imwrite(f'{out}/Bicubic.png', big[c, c])
        io.imwrite(f'{out}/overview.png',
                   cv2.resize(big, (800, 800), interpolation=cv2.INTER_AREA))
        for name, fn, _ in models:
            io.imwrite(f'{out}/{name}.png', sr_tiled(fn, lr_full)[c, c])
        meta[key] = dict(title=f'test {key[4:]} (Incheon) — no target',
                         full=[1800, 1800], overview=[800, 800],
                         origin=[ox, oy], extent=[768, 768], has_hr=False)
        print(f'{key}: {TEST_TILE[key]}')

    json.dump(meta, open(f'{res}/meta.json', 'w'), indent=1)
    with open(f'{res}/metrics.csv', 'w') as f:
        f.write('model,PSNR,SSIM,params_M\n')
        for n, p, s, par in rows:
            f.write(f'{n},{p},{s},{par}\n')
    print('meta.json / metrics.csv 갱신')


if __name__ == '__main__':
    main()
