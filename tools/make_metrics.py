"""비교 페이지용 정량 지표를 다시 만든다.

    python tools/make_metrics.py

검증 10패치 전부에 대해 배포 체크포인트 일곱 개를 돌려 두 파일을 낸다.

    results/comparison/metrics.csv            모델별 평균 (기존 파일, 덮어쓴다)
    results/comparison/metrics_per_patch.csv  패치별 원자료 (model, patch, PSNR, SSIM)

모델 호출 방식은 각 실습 페이지의 upscale 정의를 그대로 옮긴 것이다.
"""
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
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'


@torch.no_grad()
def _tensor255(net, lr):
    """EDSR 계열 — 0~255 그대로 넣는다."""
    t = torch.from_numpy(lr.transpose(2, 0, 1)).float()[None].to(DEV)
    return net(t).clamp(0, 255).round()[0].cpu().numpy().transpose(1, 2, 0).astype('uint8')


@torch.no_grad()
def _tensor01(net, lr):
    """SRGAN·ESRGAN·SwinIR — 0~1 로 정규화해 넣는다."""
    t = torch.from_numpy(lr.transpose(2, 0, 1)).float()[None].to(DEV) / 255
    return (net(t).clamp(0, 1)[0].cpu().numpy().transpose(1, 2, 0) * 255).round().astype('uint8')


def build():
    """(이름, upscale 함수, 파라미터 수 M) 목록. 실습 페이지와 같은 가중치다."""
    out = []
    n = load_srcnn(f'{CK}/01_srcnn_x3/checkpoints/srcnn_x3.pth', device=DEV)
    out.append(('SRCNN', lambda lr, n=n: srcnn_upscale(n, lr), n))
    n = load_vdsr(f'{CK}/02_vdsr_x3/checkpoints/vdsr_x3.pth', device=DEV)
    out.append(('VDSR', lambda lr, n=n: vdsr_upscale(n, lr), n))
    n = load_edsr(f'{CK}/03_edsr_x3/checkpoints/edsr_x3.pt', device=DEV)
    out.append(('EDSR', lambda lr, n=n: _tensor255(n, lr), n))
    n = load_srgan(f'{CK}/04_srgan_x3/checkpoints/srgan_g_x3.pth', device=DEV)
    out.append(('SRGAN', lambda lr, n=n: _tensor01(n, lr), n))
    n = load_esrgan(f'{CK}/05_esrgan_x3/checkpoints/esrgan_g_x3.pth', device=DEV)
    out.append(('ESRGAN', lambda lr, n=n: _tensor01(n, lr), n))
    n = load_swinir(f'{CK}/06_swinir_x3/checkpoints/swinir_x3.pth', device=DEV)
    out.append(('SwinIR', lambda lr, n=n: _tensor01(n, lr), n))
    n = load_hat(f'{CK}/07_hat_x3/checkpoints/hat_x3.pth', device=DEV)
    out.append(('HAT', lambda lr, n=n: hat_upscale(n, lr), n))
    return out


def main():
    stems = sorted(os.listdir(f'{HERE}/dataset/validation/HR'))
    stems = [s[:-4] for s in stems if s.endswith('.png')]
    print(f'검증 {len(stems)}패치, device={DEV}')

    data = []
    for s in stems:
        hr = U.imageio.imread(f'{HERE}/dataset/validation/HR/{s}.png')
        lr = U.imageio.imread(f'{HERE}/dataset/validation/LR_bicubic/X3/{s}x3.png')
        data.append((s, lr, hr))

    rows = []
    for s, lr, hr in data:
        p, q = U.score(U.bicubic(lr), hr)
        rows.append(('Bicubic', s, p, q))
    print(f'  Bicubic  {np.mean([r[2] for r in rows]):.4f}')

    sizes = {'Bicubic': ''}
    for name, fn, net in build():
        for s, lr, hr in data:
            p, q = U.score(fn(lr), hr)
            rows.append((name, s, p, q))
        sizes[name] = sum(x.numel() for x in net.parameters()) / 1e6
        mp = np.mean([r[2] for r in rows if r[0] == name])
        print(f'  {name:8s} {mp:.4f}')

    out = f'{HERE}/results/comparison'
    with open(f'{out}/metrics_per_patch.csv', 'w') as f:
        f.write('model,patch,PSNR,SSIM\n')
        for m, s, p, q in rows:
            f.write(f'{m},{s},{p},{q}\n')

    order = ['Bicubic'] + [n for n, _, _ in build()]
    with open(f'{out}/metrics.csv', 'w') as f:
        f.write('model,PSNR,SSIM,params_M\n')
        for m in order:
            v = [(p, q) for n, _, p, q in rows if n == m]
            f.write(f'{m},{np.mean([x[0] for x in v])},'
                    f'{np.mean([x[1] for x in v])},{sizes[m]}\n')
    print('saved', f'{out}/metrics_per_patch.csv')


if __name__ == '__main__':
    main()
