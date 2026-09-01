"""SR_dataset/training_ikonos -> EDSR가 읽는 DIV2K식 PNG 폴더로 변환.

기존 ikonos/ 와 다른 점:
  - 도시별 전체 모자이크 1쌍(LR 10m / HR 3.3333m)이고 이미 HR = LR x 3 이 정확히 성립
  - 실제 S2 LR 대신 이 LR 만 쓴다 (검증 없음)
  - HR 모자이크 가장자리에 nodata(검은색)가 1~12% 있어 그 타일은 버린다

    python prepare_training_ikonos.py

출력:
    sr_data/IKONOSFULL/train/HR/{city}_y####_x####.png
    sr_data/IKONOSFULL/train/LR_bicubic/X3/{city}_y####_x####x3.png
    sr_data/IKONOSFULL/val/...   <- EDSR 루프가 매 epoch test()를 부르므로
                                    타일 1장만 넣어 둔 더미. 지표는 의미 없다.
"""
import argparse
import glob
import os

import imageio.v2 as imageio
import numpy as np
import rasterio

SCALE = 3


def read_rgb(path):
    with rasterio.open(path) as src:
        return np.ascontiguousarray(src.read([1, 2, 3]).transpose(1, 2, 0))


def find_pairs(src):
    """(city, lr_path, hr_path) 목록. HR은 AOI_*/ 안, LR은 같은 stem의 모자이크."""
    pairs = []
    for hr in sorted(glob.glob(os.path.join(src, 'AOI_*', '*_8bit.tif'))):
        city = os.path.basename(os.path.dirname(hr))
        stem = os.path.basename(hr).replace('_8bit.tif', '')
        lr = os.path.join(src, f'{stem}_8bit_lr_s0_dcall-hr.tif')
        if not os.path.exists(lr):
            print(f'  [skip] {city}: LR 없음 ({os.path.basename(lr)})')
            continue
        pairs.append((city, lr, hr))
    return pairs


def tile_starts(size, tile, stride):
    if size <= tile:
        return [0]
    starts = list(range(0, size - tile + 1, stride))
    if starts[-1] != size - tile:
        starts.append(size - tile)
    return starts


def save_pair(lr, hr, stem, out_root, split):
    dir_hr = os.path.join(out_root, split, 'HR')
    dir_lr = os.path.join(out_root, split, 'LR_bicubic', f'X{SCALE}')
    os.makedirs(dir_hr, exist_ok=True)
    os.makedirs(dir_lr, exist_ok=True)
    imageio.imwrite(os.path.join(dir_hr, f'{stem}.png'), hr)
    imageio.imwrite(os.path.join(dir_lr, f'{stem}x{SCALE}.png'), lr)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--src', default='../SR_dataset/training_ikonos')
    p.add_argument('--out', default='../sr_data/IKONOSFULL')
    p.add_argument('--tile', type=int, default=128, help='학습 타일 크기(LR 픽셀)')
    p.add_argument('--stride', type=int, default=128)
    args = p.parse_args()

    pairs = find_pairs(args.src)
    if not pairs:
        raise SystemExit(f'쌍을 찾지 못했습니다: {args.src}')

    n_keep = n_drop = 0
    first = None
    for city, lr_p, hr_p in pairs:
        lr, hr = read_rgb(lr_p), read_rgb(hr_p)
        h, w = lr.shape[:2]
        if hr.shape[:2] != (h * SCALE, w * SCALE):
            raise SystemExit(f'{city}: HR {hr.shape[:2]} != 3x LR {(h * 3, w * 3)}')

        keep = drop = 0
        for y in tile_starts(h, args.tile, args.stride):
            for x in tile_starts(w, args.tile, args.stride):
                t = args.tile
                lt = lr[y:y + t, x:x + t]
                ht = hr[y * SCALE:(y + t) * SCALE, x * SCALE:(x + t) * SCALE]
                # nodata(3채널 모두 0)가 하나라도 있으면 버린다
                if (ht.sum(2) == 0).any() or (lt.sum(2) == 0).any():
                    drop += 1
                    continue
                stem = f'{city}_y{y:05d}_x{x:05d}'
                save_pair(lt, ht, stem, args.out, 'train')
                if first is None:
                    first = (lt, ht, stem)
                keep += 1
        n_keep += keep
        n_drop += drop
        print(f'  {city:14s} LR {w}x{h} -> 타일 {keep}개 (nodata로 버림 {drop}개)')

    # EDSR main loop 가 매 epoch test() 를 부르므로 더미 val 1장
    save_pair(*first, args.out, 'val')
    print(f'\n완료: train 타일 {n_keep}개 (버림 {n_drop}) -> {args.out}')
    print(f'      val 은 더미 1장({first[2]}). 검증 안 하므로 지표 무시.')


if __name__ == '__main__':
    main()
