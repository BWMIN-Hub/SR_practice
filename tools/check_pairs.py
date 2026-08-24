"""HR-LR 쌍 품질 검사 — 패치별 차이를 재고 이상치를 찾는다.

    python tools/check_pairs.py

training 은 HR vs g_LR(합성), validation 은 HR vs LR(실제 S2) 을 비교한다.
g_LR 은 HR 에서 파생됐으므로 방사가 일치하는 것이 정상이고, 여기서 이상치가 나오면
전처리 문제를 의심해야 한다. 실제 S2 쪽은 밝기 오프셋이 큰 것이 정상이다.

주의: MAE 가 큰 패치를 불량으로 오해하지 말 것. MAE 는 텍스처량과 상관이 +0.63 이라
      MAE 상위는 어긋난 쌍이 아니라 디테일이 많은 패치다.
"""
import glob, os
import numpy as np
import cv2
import imageio.v2 as imageio

ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'dataset')


def hist_dist(a, b, bins=64):
    """채널별 정규화 히스토그램의 Bhattacharyya 거리 평균 (0=동일, 1=완전히 다름)."""
    ds = []
    for c in range(3):
        ha = np.histogram(a[..., c], bins=bins, range=(0, 256), density=True)[0]
        hb = np.histogram(b[..., c], bins=bins, range=(0, 256), density=True)[0]
        ha, hb = ha / (ha.sum() + 1e-12), hb / (hb.sum() + 1e-12)
        bc = np.sqrt(ha * hb).sum()               # Bhattacharyya coefficient
        ds.append(np.sqrt(max(0.0, 1.0 - bc)))
    return float(np.mean(ds))


def measure(split):
    rows = []
    for f in sorted(glob.glob(f'{ROOT}/{split}/HR/*.png')):
        stem = os.path.basename(f)[:-4]
        hr = imageio.imread(f).astype(np.float32)
        lr = imageio.imread(f'{ROOT}/{split}/LR_bicubic/X3/{stem}x3.png').astype(np.float32)
        up = cv2.resize(lr, (hr.shape[1], hr.shape[0]), interpolation=cv2.INTER_CUBIC)

        rows.append(dict(
            stem=stem,
            d_mean=float(hr.mean() - lr.mean()),                  # 밝기 오프셋
            d_std=float(hr.std() - lr.std()),                     # 대비 차이
            mae=float(np.abs(hr - up).mean()),                    # 화소 차이
            corr=float(np.corrcoef(hr.ravel(), up.ravel())[0, 1]),
            hdist=hist_dist(hr, lr),                              # 히스토그램 거리
        ))
    return rows


for split in ['training', 'validation']:
    rows = measure(split)
    src = 'g_LR (합성)' if split == 'training' else 'LR (실제 S2)'
    print(f'\n{"="*78}\n{split}  —  HR vs {src},  패치 {len(rows)}개\n{"="*78}')
    for k, unit in [('d_mean', 'DN'), ('d_std', 'DN'), ('mae', 'DN'),
                    ('corr', ''), ('hdist', '')]:
        v = np.array([r[k] for r in rows])
        print(f'  {k:7s} 평균 {v.mean():8.3f}  표준편차 {v.std():7.3f}  '
              f'최소 {v.min():8.3f}  최대 {v.max():8.3f} {unit}')

    # 이상치: 중앙값절대편차(MAD) 기준 — 정규분포 가정을 피한다
    print(f'\n  {"패치":42s} {"Δmean":>8s} {"MAE":>7s} {"corr":>6s} {"hdist":>7s}')
    h = np.array([r['hdist'] for r in rows])
    med, mad = np.median(h), np.median(np.abs(h - np.median(h)))
    thr = med + 3 * 1.4826 * mad
    for r in sorted(rows, key=lambda r: -r['hdist'])[:6]:
        flag = ' <== 이상치' if r['hdist'] > thr else ''
        print(f'  {r["stem"][:42]:42s} {r["d_mean"]:8.2f} {r["mae"]:7.2f} '
              f'{r["corr"]:6.3f} {r["hdist"]:7.4f}{flag}')
    print(f'  (hdist 이상치 임계 = 중앙값 {med:.4f} + 3·MAD = {thr:.4f})')
