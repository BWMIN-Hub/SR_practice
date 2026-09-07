"""실습 데이터셋을 v2 페어(g_LR / g_HR)로 다시 만든다.

패치 이름과 자르는 위치는 그대로 두고 출처만 바꾼다. 그래야 각 페이지의 확대
좌표와 대표 패치가 계속 맞는다.

  기존: SR_dataset/ikonos 의 g_LR / HR
  이번: SR_dataset/v2_training/ikonos_v2_hr_S0050 의 glr / ghr

v2 는 입력과 정답의 방사가 서로 맞아 있다. 그래서 이 쌍으로 학습한 가중치를
예전 쌍으로 재면 손해가 나고, 반대도 마찬가지다. 재는 쪽과 학습한 쪽을 맞춰야 한다.

    python tools/rebuild_dataset_v2.py
"""
import argparse
import os
import re
import shutil

import numpy as np
import rasterio

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = '/c/work/nst_work_dirs/G2601-SAR-Filtering/tmp_files'
SCALE = 3
TILE = 128


def rd(p):
    with rasterio.open(p) as r:
        return np.transpose(r.read([1, 2, 3]), (1, 2, 0))


def main():
    import cv2
    ap = argparse.ArgumentParser()
    ap.add_argument('--variant', default='S0050', help='v2 변형 중 어느 것을 쓸지')
    ap.add_argument('--dry', action='store_true')
    a = ap.parse_args()

    src = f'{ROOT}/SR_dataset/v2_training/ikonos_v2_hr_{a.variant}'
    cache = {}
    n = {'training': 0, 'validation': 0}
    miss = []
    for split in ('training', 'validation'):
        d_hr = f'{HERE}/dataset/{split}/HR'
        d_lr = f'{HERE}/dataset/{split}/LR_bicubic/X{SCALE}'
        stems = sorted(f[:-4] for f in os.listdir(d_hr) if f.endswith('.png'))
        for s in stems:
            m = re.match(r'(.+)_y(\d+)_x(\d+)$', s)
            sc, y, x = m.group(1), int(m.group(2)), int(m.group(3))
            if sc not in cache:
                p = f'{src}/{sc}_ghr.tif'
                if not os.path.exists(p):
                    miss.append(s); continue
                cache[sc] = (rd(p), rd(f'{src}/{sc}_glr.tif'))
            ghr, glr = cache[sc]
            if y + TILE > glr.shape[0] or x + TILE > glr.shape[1]:
                miss.append(s); continue
            hr = ghr[y * SCALE:(y + TILE) * SCALE, x * SCALE:(x + TILE) * SCALE]
            lr = glr[y:y + TILE, x:x + TILE]
            if not a.dry:
                cv2.imwrite(f'{d_hr}/{s}.png', hr[:, :, ::-1])
                cv2.imwrite(f'{d_lr}/{s}x{SCALE}.png', lr[:, :, ::-1])
            n[split] += 1
        print(f'{split}: {n[split]}장 교체')
    if miss:
        print('건너뜀:', ', '.join(miss))
    print(f'출처 {src}')


if __name__ == '__main__':
    main()
