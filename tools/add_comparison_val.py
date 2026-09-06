"""비교 자산에 검증 패치를 더 넣는다 (val3, val4 ...).

기존 val1/val2 와 같은 형식으로 저장한다.
  results/comparison/<key>/{LR,Bicubic,HR,overview,모델7}.png
  results/comparison/meta.json 에 항목 추가

    python tools/add_comparison_val.py --add val3=AOI_Busan_7_y0064_x0192 \
                                       --add val4=AOI_Barcelona_2_y0384_x0064
"""
import argparse
import json
import os
import sys

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
DEV = os.environ.get('IQA_DEV', 'cuda')
SCALE = 3


@torch.no_grad()
def t255(net, lr):
    t = torch.from_numpy(lr.transpose(2, 0, 1)).float()[None].to(DEV)
    return net(t).clamp(0, 255).round()[0].cpu().numpy().transpose(1, 2, 0).astype('uint8')


@torch.no_grad()
def t01(net, lr):
    t = torch.from_numpy(lr.transpose(2, 0, 1)).float()[None].to(DEV) / 255
    return (net(t).clamp(0, 1)[0].cpu().numpy().transpose(1, 2, 0) * 255).round().astype('uint8')


def build():
    """실습 페이지와 같은 가중치·같은 호출 방식."""
    out = []
    n = load_srcnn(f'{CK}/01_srcnn_x3/checkpoints/srcnn_x3.pth', device=DEV)
    out.append(('SRCNN', lambda lr, n=n: srcnn_upscale(n, lr)))
    n = load_vdsr(f'{CK}/02_vdsr_x3/checkpoints/vdsr_x3.pth', device=DEV)
    out.append(('VDSR', lambda lr, n=n: vdsr_upscale(n, lr)))
    n = load_edsr(f'{CK}/03_edsr_x3/checkpoints/edsr_x3.pt', device=DEV)
    out.append(('EDSR', lambda lr, n=n: t255(n, lr)))
    n = load_srgan(f'{CK}/04_srgan_x3/checkpoints/srgan_g_x3.pth', device=DEV)
    out.append(('SRGAN', lambda lr, n=n: t01(n, lr)))
    n = load_esrgan(f'{CK}/05_esrgan_x3/checkpoints/esrgan_g_x3.pth', device=DEV)
    out.append(('ESRGAN', lambda lr, n=n: t01(n, lr)))
    n = load_swinir(f'{CK}/06_swinir_x3/checkpoints/swinir_x3.pth', device=DEV)
    out.append(('SwinIR', lambda lr, n=n: t01(n, lr)))
    n = load_hat(f'{CK}/07_hat_x3/checkpoints/hat_x3.pth', device=DEV)
    out.append(('HAT', lambda lr, n=n: hat_upscale(n, lr)))
    return out


def main():
    import imageio.v2 as io
    ap = argparse.ArgumentParser()
    ap.add_argument('--add', action='append', required=True,
                    help='"key=stem" 형식. 예: val3=AOI_Busan_7_y0064_x0192')
    a = ap.parse_args()

    res = f'{HERE}/results/comparison'
    meta = json.load(open(f'{res}/meta.json'))
    models = build()

    for spec in a.add:
        key, stem = spec.split('=', 1)
        d = f'{HERE}/dataset/validation'
        lr = io.imread(f'{d}/LR_bicubic/X{SCALE}/{stem}x{SCALE}.png')
        hr = io.imread(f'{d}/HR/{stem}.png')
        out = f'{res}/{key}'
        os.makedirs(out, exist_ok=True)

        io.imwrite(f'{out}/LR.png', U.nearest(lr, SCALE))
        io.imwrite(f'{out}/Bicubic.png', U.bicubic(lr, SCALE))
        io.imwrite(f'{out}/HR.png', hr)
        io.imwrite(f'{out}/overview.png', hr)
        for name, fn in models:
            io.imwrite(f'{out}/{name}.png', fn(lr))
        h, w = hr.shape[:2]
        meta[key] = dict(title=f'{key.replace("val", "validation ")} — {stem}',
                         full=[h, w], overview=[h, w], origin=[0, 0],
                         extent=[w, h], has_hr=True)
        p, s = U.score(U.bicubic(lr, SCALE), hr)
        print(f'{key}: {stem}  bicubic {p:.2f} / {s:.4f}  -> {out}')

    json.dump(meta, open(f'{res}/meta.json', 'w'), indent=1)
    print('meta.json 갱신:', ', '.join(meta))


if __name__ == '__main__':
    main()
