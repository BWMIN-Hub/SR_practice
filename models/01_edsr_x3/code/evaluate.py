"""검증 씬(실제 S2 LR -> HR)에서 Bicubic 대비 EDSR 성능을 재고 비교 그림을 만든다.

    python evaluate.py --weight experiment/edsr_s2_x3/model/model_best.pt

출력:
    results/val/metrics.csv          씬별 PSNR/SSIM (Bicubic vs EDSR)
    results/val/{scene}_compare.png  LR(bicubic) | EDSR | HR 확대 비교
    results/val/{scene}_SRx3.png     전체 씬 SR 결과
"""
import argparse
import glob
import os

import cv2
import imageio.v2 as imageio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import torch
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

from infer import build_model, sr_tiled, write_geotiff


def read_rgb(path):
    with rasterio.open(path) as src:
        return np.ascontiguousarray(src.read([1, 2, 3]).transpose(1, 2, 0))


def radio_match(pred, gt, shave):
    """pred를 gt에 채널별 선형(gain/offset)으로 맞춘다.

    센서/촬영일이 달라 생기는 밝기 차이는 SR 품질과 무관한데 PSNR을 지배한다.
    bicubic과 EDSR에 똑같이 적용하므로 둘 사이 비교는 공정하고,
    '기하/디테일만 놓고 보면 얼마인가'를 보는 진단용 수치다(GT 통계를 쓰므로 oracle).
    """
    out = np.empty_like(pred, dtype=np.float32)
    p = pred[shave:-shave, shave:-shave].astype(np.float32)
    g = gt[shave:-shave, shave:-shave].astype(np.float32)
    for c in range(3):
        x, y = p[..., c].ravel(), g[..., c].ravel()
        # polyfit은 0~255 raw 값에서 조건수가 나빠 경고를 낸다. 중심화한
        # 닫힌 형태 최소제곱(1차)은 같은 해를 안정적으로 준다.
        xm, ym = x.mean(), y.mean()
        var = ((x - xm) ** 2).mean()
        a = ((x - xm) * (y - ym)).mean() / var if var > 1e-8 else 1.0
        b = ym - a * xm
        out[..., c] = pred[..., c].astype(np.float32) * a + b
    return out.clip(0, 255).round().astype(np.uint8)


def score(pred, gt, shave):
    """경계 shave 픽셀을 잘라내고 PSNR/SSIM 계산."""
    p = pred[shave:-shave, shave:-shave]
    g = gt[shave:-shave, shave:-shave]
    return (
        psnr(g, p, data_range=255),
        ssim(g, p, data_range=255, channel_axis=2),
    )


def crop_figure(bic, sr, hr, out_path, scene, size=200):
    """가장 디테일이 많은 영역을 골라 3분할 비교 그림 저장."""
    h, w = hr.shape[:2]
    # 분산이 큰(=텍스처가 많은) 위치를 대충 고른다
    gray = cv2.cvtColor(hr, cv2.COLOR_RGB2GRAY)
    step = size
    best, best_yx = -1, (0, 0)
    for y in range(0, h - size, step):
        for x in range(0, w - size, step):
            v = gray[y:y + size, x:x + size].std()
            if v > best:
                best, best_yx = v, (y, x)
    y, x = best_yx

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.6))
    for ax, img, title in zip(
        axes,
        [bic, sr, hr],
        ['Bicubic x3', 'EDSR x3', 'HR (ground truth)'],
    ):
        ax.imshow(img[y:y + size, x:x + size])
        ax.set_title(title, fontsize=12)
        ax.axis('off')
    fig.suptitle(f'{scene}  (crop {size}x{size} @ y={y}, x={x})', fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches='tight')
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--weight', default='experiment/edsr_s2_x3/model/model_best.pt')
    p.add_argument('--src', default='../SR_dataset')
    p.add_argument('--output', default='results/val')
    p.add_argument('--scenes', default='',
                   help='평가할 씬 이름(쉼표 구분) 또는 씬 이름이 줄마다 든 파일. '
                        '안 주면 --src 아래 전부')
    p.add_argument('--lr_name', default='LR', choices=('LR', 'g_LR'),
                   help='입력 LR 소스. LR=실제 S2(검증), g_LR=합성 LR(학습과 동일 도메인)')
    p.add_argument('--scale', type=int, default=3)
    p.add_argument('--n_resblocks', type=int, default=16)
    p.add_argument('--n_feats', type=int, default=64)
    p.add_argument('--res_scale', type=float, default=1.0)
    p.add_argument('--tile', type=int, default=256)
    p.add_argument('--pad', type=int, default=16)
    p.add_argument('--radio_match', action='store_true',
                   help='방사 정규화 후 지표도 함께 출력(밝기 차이 제거, 진단용)')
    p.add_argument('--png', action='store_true', help='GeoTIFF 외에 전체 씬 PNG도 저장')
    p.add_argument('--cpu', action='store_true')
    args = p.parse_args()

    device = torch.device('cpu' if args.cpu else 'cuda')
    net = build_model(args, device)
    os.makedirs(args.output, exist_ok=True)

    scenes = sorted(
        os.path.basename(os.path.dirname(f))
        for f in glob.glob(os.path.join(args.src, '*', f'{args.lr_name}.tif'))
    )
    if args.scenes:
        if os.path.isfile(args.scenes):
            keep = {l.strip() for l in open(args.scenes) if l.strip()}
        else:
            keep = {v.strip() for v in args.scenes.split(',') if v.strip()}
        scenes = [s for s in scenes if s in keep]
    if not scenes:
        raise SystemExit(f'{args.lr_name}.tif 를 가진 씬이 없습니다: {args.src}')
    print(f'입력 LR = {args.lr_name}.tif, 씬 {len(scenes)}개\n')

    shave = args.scale + 6
    rows, mrows = [], []
    for scene in scenes:
        d = os.path.join(args.src, scene)
        lr_path = os.path.join(d, f'{args.lr_name}.tif')
        lr = read_rgb(lr_path)
        hr = read_rgb(os.path.join(d, 'HR.tif'))
        h, w = lr.shape[:2]
        hr = hr[: h * args.scale, : w * args.scale]

        bic = cv2.resize(lr, (w * args.scale, h * args.scale),
                         interpolation=cv2.INTER_CUBIC)
        sr = sr_tiled(net, lr, args.scale, args.tile, args.pad, device)

        p_b, s_b = score(bic, hr, shave)
        p_s, s_s = score(sr, hr, shave)
        rows.append((scene, p_b, s_b, p_s, s_s))
        line = (f'{scene:18s} bicubic {p_b:6.3f}dB/{s_b:.4f}   '
                f'EDSR {p_s:6.3f}dB/{s_s:.4f}   ({p_s - p_b:+.3f}dB)')
        if args.radio_match:
            mb, _ = score(radio_match(bic, hr, shave), hr, shave)
            ms, _ = score(radio_match(sr, hr, shave), hr, shave)
            mrows.append((scene, mb, ms))
            line += f'   | 방사정규화 bicubic {mb:6.3f} EDSR {ms:6.3f} ({ms - mb:+.3f}dB)'
        print(line)

        tag = args.lr_name
        # 좌표 붙은 GeoTIFF (QGIS 등에서 HR/LR과 겹쳐볼 수 있음)
        write_geotiff(os.path.join(args.output, f'{scene}_{tag}_SRx{args.scale}.tif'),
                      sr, lr_path, args.scale)
        write_geotiff(os.path.join(args.output, f'{scene}_{tag}_bicubicx{args.scale}.tif'),
                      bic, lr_path, args.scale)
        if args.png:
            imageio.imwrite(
                os.path.join(args.output, f'{scene}_{tag}_SRx{args.scale}.png'), sr)
        crop_figure(bic, sr, hr,
                    os.path.join(args.output, f'{scene}_{tag}_compare.png'),
                    f'{scene}  (input: {tag}.tif)')

    csv_path = os.path.join(args.output, f'metrics_{args.lr_name}.csv')
    with open(csv_path, 'w') as f:
        f.write('scene,psnr_bicubic,ssim_bicubic,psnr_edsr,ssim_edsr,psnr_gain\n')
        for r in rows:
            f.write(f'{r[0]},{r[1]:.4f},{r[2]:.5f},{r[3]:.4f},{r[4]:.5f},'
                    f'{r[3] - r[1]:.4f}\n')
        m = np.array([[r[1], r[2], r[3], r[4]] for r in rows]).mean(axis=0)
        f.write(f'MEAN,{m[0]:.4f},{m[1]:.5f},{m[2]:.4f},{m[3]:.5f},'
                f'{m[2] - m[0]:.4f}\n')

    print(f'\n평균  bicubic {m[0]:.3f}dB/{m[1]:.4f}   '
          f'EDSR {m[2]:.3f}dB/{m[3]:.4f}   ({m[2] - m[0]:+.3f}dB)')
    if mrows:
        mm = np.array([[r[1], r[2]] for r in mrows]).mean(axis=0)
        print(f'방사정규화 평균  bicubic {mm[0]:.3f}dB   EDSR {mm[1]:.3f}dB   '
              f'({mm[1] - mm[0]:+.3f}dB)')
    print(f'-> {csv_path}')


if __name__ == '__main__':
    main()
