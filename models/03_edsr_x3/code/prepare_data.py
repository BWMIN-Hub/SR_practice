"""SR_dataset(GeoTIFF) -> EDSR가 읽는 DIV2K식 PNG 폴더 구조로 변환.

    sr_data/S2SR/
      train/HR/{scene}_y00_x00.png              <- HR.tif   타일  (GT)
      train/LR_bicubic/X3/{scene}_y00_x00x3.png <- g_LR.tif 타일  (합성 LR)
      val/HR/{scene}.png                        <- HR.tif   전체 씬
      val/LR_bicubic/X3/{scene}x3.png           <- LR.tif   전체 씬 (실제 S2)

train은 g_LR을 가진 씬 전부, val은 LR.tif를 가진 씬만 만든다.
--val_scenes 를 주면 그 씬들은 학습에서 빼고 검증 전용으로 쓴다(씬 단위 holdout).

학습셋을 타일로 자르는 이유:
  EDSR의 SRData.__getitem__ 은 패치 하나를 뽑을 때마다 이미지 '전체'를
  unpickle 한다. 3000x3000 HR 을 통째로 두면 샘플당 30MB 를 읽게 되어
  epoch 당 수백 GB 를 역직렬화하게 되고 GPU가 놀아버린다.
  미리 타일로 잘라두면 샘플당 수백 KB 로 줄어 학습이 GPU-bound 가 된다.
  val 은 씬 단위 PSNR 을 봐야 하므로 자르지 않는다.
"""
import argparse
import os

import imageio.v2 as imageio
import numpy as np
import rasterio

SCALE = 3


def read_rgb(path):
    """GeoTIFF를 (H, W, 3) uint8로 읽는다."""
    with rasterio.open(path) as src:
        arr = src.read([1, 2, 3])  # (3, H, W)
    return np.ascontiguousarray(arr.transpose(1, 2, 0))


def load_pair(hr_tif, lr_tif, scene):
    """HR을 LR의 정확히 SCALE배로 맞춰 반환."""
    hr = read_rgb(hr_tif)
    lr = read_rgb(lr_tif)
    h, w = lr.shape[:2]
    hr = hr[: h * SCALE, : w * SCALE]
    if hr.shape[0] != h * SCALE or hr.shape[1] != w * SCALE:
        raise ValueError(
            f'{scene}: HR {hr.shape[:2]} != {SCALE}x LR {(h * SCALE, w * SCALE)}'
        )
    return lr, hr


def save_pair(lr, hr, stem, out_root, split):
    dir_hr = os.path.join(out_root, split, 'HR')
    dir_lr = os.path.join(out_root, split, 'LR_bicubic', f'X{SCALE}')
    os.makedirs(dir_hr, exist_ok=True)
    os.makedirs(dir_lr, exist_ok=True)
    imageio.imwrite(os.path.join(dir_hr, f'{stem}.png'), hr)
    imageio.imwrite(os.path.join(dir_lr, f'{stem}x{SCALE}.png'), lr)


def tile_starts(size, tile, stride):
    """마지막 타일은 경계에 맞춰 당겨서 잘림 없이 덮는다."""
    if tile <= 0 or size <= tile:
        return [0]
    starts = list(range(0, size - tile + 1, stride))
    if starts[-1] != size - tile:
        starts.append(size - tile)
    return starts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', default='../SR_dataset',
                        help='HR.tif / g_LR.tif / LR.tif 가 들어있는 씬 폴더들의 상위 경로')
    parser.add_argument('--out', default='../sr_data/S2SR',
                        help='생성할 데이터셋 루트')
    parser.add_argument('--tile', type=int, default=128,
                        help='학습 타일 크기 (LR 픽셀). 0이면 자르지 않음')
    parser.add_argument('--stride', type=int, default=128,
                        help='학습 타일 stride (LR 픽셀)')
    parser.add_argument('--val_scenes', default='',
                        help='검증으로 뺄 씬 이름(쉼표 구분). 주면 그 씬은 학습에서 제외하고 '
                             '검증에만 쓴다. 안 주면 g_LR 있는 씬 전부 학습 + LR 있는 씬 전부 검증')
    args = parser.parse_args()

    holdout = {v.strip() for v in args.val_scenes.split(',') if v.strip()}

    scenes = sorted(
        d for d in os.listdir(args.src)
        if os.path.isdir(os.path.join(args.src, d))
    )
    if not scenes:
        raise SystemExit(f'씬을 찾지 못했습니다: {args.src}')

    n_tiles = n_val = 0
    for scene in scenes:
        d = os.path.join(args.src, scene)
        hr_tif = os.path.join(d, 'HR.tif')
        if not os.path.exists(hr_tif):
            print(f'  [skip] {scene}: HR.tif 없음')
            continue

        # ---- 학습: 합성 LR(g_LR) + 타일링 ----
        g_lr = os.path.join(d, 'g_LR.tif')
        if os.path.exists(g_lr) and scene not in holdout:
            lr, hr = load_pair(hr_tif, g_lr, scene)
            h, w = lr.shape[:2]
            ys = tile_starts(h, args.tile, args.stride)
            xs = tile_starts(w, args.tile, args.stride)
            t = args.tile if args.tile > 0 else max(h, w)
            cnt = 0
            for y in ys:
                for x in xs:
                    lt = lr[y:y + t, x:x + t]
                    ht = hr[y * SCALE:(y + t) * SCALE, x * SCALE:(x + t) * SCALE]
                    save_pair(lt, ht, f'{scene}_y{y:04d}_x{x:04d}',
                              args.out, 'train')
                    cnt += 1
            n_tiles += cnt
            print(f'  [train] {scene:18s} LR {w}x{h} -> {cnt} tiles '
                  f'({t}x{t} LR / {t * SCALE}x{t * SCALE} HR)')

        # ---- 검증: 실제 S2 LR + 전체 씬 ----
        lr_tif = os.path.join(d, 'LR.tif')
        if os.path.exists(lr_tif) and (not holdout or scene in holdout):
            lr, hr = load_pair(hr_tif, lr_tif, scene)
            save_pair(lr, hr, scene, args.out, 'val')
            n_val += 1
            print(f'  [val]   {scene:18s} LR {lr.shape[1]}x{lr.shape[0]} '
                  f'-> HR {hr.shape[1]}x{hr.shape[0]} (전체 씬)')

    print(f'\n완료: train 타일 {n_tiles}개 / val 씬 {n_val}개 -> {args.out}')


if __name__ == '__main__':
    main()
