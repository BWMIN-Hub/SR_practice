"""학습한 EDSR로 전체 씬을 x3 SR 하고 GeoTIFF + PNG로 저장.

    python infer.py \
        --weight experiment/edsr_s2_x3/model/model_best.pt \
        --input  ../SR_testdataset \
        --output results/test

지오리퍼런스는 입력에서 가져와 픽셀 크기만 1/scale 로 줄여 기록한다.
큰 씬도 처리되도록 겹침(overlap) 타일링으로 추론한다.
"""
import argparse
import glob
import os
import sys
import types

import imageio.v2 as imageio
import numpy as np
import rasterio
import torch
from rasterio.transform import Affine

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))
from model import edsr  # noqa: E402


def build_model(args, device):
    margs = types.SimpleNamespace(
        n_resblocks=args.n_resblocks,
        n_feats=args.n_feats,
        scale=[args.scale],
        rgb_range=255,
        n_colors=3,
        res_scale=args.res_scale,
    )
    net = edsr.make_model(margs)
    state = torch.load(args.weight, map_location='cpu')
    net.load_state_dict(state, strict=False)
    return net.eval().to(device)


def write_geotiff(path, arr, ref_path, scale):
    """arr(H*scale, W*scale, 3)를 ref_path(LR GeoTIFF)의 좌표계에 맞춰 저장.

    지리 범위와 CRS는 그대로 두고 픽셀 크기만 1/scale 로 줄인다.
    따라서 출력 GeoTIFF는 입력과 정확히 같은 영역을 덮는다.
    """
    with rasterio.open(ref_path) as src:
        profile = src.profile
        transform = src.transform

    profile.update(
        driver='GTiff',
        height=arr.shape[0],
        width=arr.shape[1],
        count=3,
        dtype='uint8',
        transform=transform * Affine.scale(1 / scale, 1 / scale),
        compress='deflate',
    )
    with rasterio.open(path, 'w', **profile) as dst:
        dst.write(arr.transpose(2, 0, 1))


@torch.no_grad()
def sr_tiled(net, lr, scale, tile, pad, device):
    """lr: (H, W, 3) uint8 -> (H*scale, W*scale, 3) uint8"""
    h, w = lr.shape[:2]
    out = np.zeros((h * scale, w * scale, 3), dtype=np.uint8)

    for y0 in range(0, h, tile):
        for x0 in range(0, w, tile):
            y1, x1 = min(y0 + tile, h), min(x0 + tile, w)

            # 이음매를 없애기 위해 pad 만큼 넓게 읽는다
            py0, px0 = max(y0 - pad, 0), max(x0 - pad, 0)
            py1, px1 = min(y1 + pad, h), min(x1 + pad, w)

            patch = lr[py0:py1, px0:px1]
            t = torch.from_numpy(patch.transpose(2, 0, 1)).float()
            t = t.unsqueeze(0).to(device)

            sr = net(t)
            sr = sr.clamp(0, 255).round().squeeze(0).cpu().numpy()
            sr = sr.transpose(1, 2, 0).astype(np.uint8)

            # 넓게 읽은 만큼 다시 잘라낸다 (모두 HR 좌표계)
            cy, cx = (y0 - py0) * scale, (x0 - px0) * scale
            out[y0 * scale:y1 * scale, x0 * scale:x1 * scale] = \
                sr[cy:cy + (y1 - y0) * scale, cx:cx + (x1 - x0) * scale]

    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--weight', default='experiment/edsr_s2_x3/model/model_best.pt')
    p.add_argument('--input', default='../SR_testdataset',
                   help='GeoTIFF 파일 또는 폴더')
    p.add_argument('--output', default='results/test')
    p.add_argument('--scale', type=int, default=3)
    p.add_argument('--n_resblocks', type=int, default=16)
    p.add_argument('--n_feats', type=int, default=64)
    p.add_argument('--res_scale', type=float, default=1.0)
    p.add_argument('--tile', type=int, default=256, help='LR 기준 타일 크기')
    p.add_argument('--pad', type=int, default=16, help='LR 기준 타일 겹침')
    p.add_argument('--png', action='store_true', help='GeoTIFF 외에 PNG도 저장')
    p.add_argument('--cpu', action='store_true')
    args = p.parse_args()

    device = torch.device('cpu' if args.cpu else 'cuda')
    net = build_model(args, device)
    os.makedirs(args.output, exist_ok=True)

    if os.path.isdir(args.input):
        files = sorted(glob.glob(os.path.join(args.input, '*.tif')))
    else:
        files = [args.input]
    if not files:
        raise SystemExit(f'입력 파일이 없습니다: {args.input}')

    for f in files:
        name = os.path.splitext(os.path.basename(f))[0]
        with rasterio.open(f) as src:
            lr = np.ascontiguousarray(src.read([1, 2, 3]).transpose(1, 2, 0))

        sr = sr_tiled(net, lr, args.scale, args.tile, args.pad, device)

        tif_path = os.path.join(args.output, f'{name}_SRx{args.scale}.tif')
        write_geotiff(tif_path, sr, f, args.scale)

        if args.png:
            imageio.imwrite(
                os.path.join(args.output, f'{name}_SRx{args.scale}.png'), sr)
        print(f'{name}: {lr.shape[1]}x{lr.shape[0]} -> {sr.shape[1]}x{sr.shape[0]}  {tif_path}')

    print(f'\n완료: {len(files)}개 -> {args.output}')


if __name__ == '__main__':
    main()
