"""실습용 데이터 로드·시각화·평가 헬퍼. `from sr_utils import *` 로 쓴다."""
import json
import os
import urllib.request

import cv2
import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as _psnr
from skimage.metrics import structural_similarity as _ssim

BASE = 'https://raw.githubusercontent.com/BWMIN-Hub/SR_practice/main'
API = 'https://api.github.com/repos/BWMIN-Hub/SR_practice/contents'

# 각 split 의 대표 패치. REP 은 한 장(호환용), REPS 는 실습에서 보여줄 두 장이다.
REP = {
    'training': 'AOI_Barcelona_10_y0128_x0128',
    'validation': 'AOI_Paris_1_6_y0064_x0192',   # 파리
}
REPS = {
    'training': ['AOI_Barcelona_10_y0128_x0128', 'AOI_Seoul_14_y0256_x0128'],
    'validation': ['AOI_Paris_1_6_y0064_x0192',   # 파리
                   'AOI_Seoul_14_y0256_x0128'],  # 서울
}
TEST = 'incheon_600.png'                          # 인천, 실제 촬영본
TESTS = ['incheon_600.png', 'incheon2_600.png']   # 같은 씬의 다른 두 구역
SHAVE = 4                                         # 점수 잴 때 잘라낼 가장자리

__all__ = ['BASE', 'REP', 'REPS', 'TEST', 'TESTS', 'SHAVE', 'fetch', 'pair', 'load_test',
           'list_split', 'show', 'zoom', 'score', 'compare', 'bicubic', 'nearest', 'retarget',
           'show_data', 'show_results', 'show_test',
           'np', 'plt', 'cv2', 'imageio', 'json', 'os', 'urllib']


def fetch(url, path):
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        urllib.request.urlretrieve(url, path)
    return path


def pair(split, stem):
    """(입력 LR, 정답 HR) 한 쌍."""
    hr = imageio.imread(fetch(f'{BASE}/dataset/{split}/HR/{stem}.png', f'{split}/{stem}.png'))
    lr = imageio.imread(fetch(f'{BASE}/dataset/{split}/LR_bicubic/X3/{stem}x3.png',
                              f'{split}/{stem}_lr.png'))
    return lr, hr


def load_test(i=0):
    """test 대표(인천) 입력. 정답은 없다. i 로 두 구역 중 하나를 고른다."""
    name = TESTS[i] if isinstance(i, int) else i
    return imageio.imread(fetch(f'{BASE}/dataset/test/{name}', f'test_{name}'))


def list_split(split):
    with urllib.request.urlopen(f'{API}/dataset/{split}/HR') as r:
        return sorted(x['name'][:-4] for x in json.load(r))


def bicubic(lr, scale=3):
    h, w = lr.shape[:2]
    return cv2.resize(lr, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)


def nearest(lr, scale=3):
    """원본 LR 을 확대해서 보여주기 위한 것. 최근접이라 화소가 그대로 각져 보인다."""
    h, w = lr.shape[:2]
    return cv2.resize(lr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_NEAREST)


def score(pred, gt):
    """(PSNR, SSIM). 가장자리는 잘라내고 잰다."""
    a, b = pred[SHAVE:-SHAVE, SHAVE:-SHAVE], gt[SHAVE:-SHAVE, SHAVE:-SHAVE]
    return _psnr(b, a, data_range=255), _ssim(b, a, data_range=255, channel_axis=2)


def show(items, title=''):
    """items: [(이름, 입력, 정답 또는 None)] — 위에 입력, 아래에 정답."""
    fig, ax = plt.subplots(2, len(items), figsize=(3.3 * len(items), 7.0), squeeze=False)
    for c, (name, lo, hi) in enumerate(items):
        ax[0, c].imshow(lo)
        ax[0, c].set_title(f'{name}\ninput {lo.shape[0]}px', fontsize=9)
        if hi is None:
            ax[1, c].text(.5, .5, 'no target', ha='center', va='center', color='#888')
            ax[1, c].set_facecolor('#f2f2f2')
        else:
            ax[1, c].imshow(hi)
            ax[1, c].set_title(f'target {hi.shape[0]}px', fontsize=9)
        for r in (0, 1):
            ax[r, c].set_xticks([]); ax[r, c].set_yticks([])
    if title:
        fig.suptitle(title, fontsize=10)
    plt.tight_layout(); plt.show()


def zoom(panels, size=110, title='', ref=None):
    """결과를 두 줄로 보여준다. panels: [(이름, 이미지)].

      윗줄 = 패치 전체 (노란 네모가 아랫줄에서 확대한 자리)
      아랫줄 = 그 구역만 확대

    확대할 자리는 가장 복잡한(경계가 많은) 구역을 자동으로 고른다.
    ref 를 주면 그 이미지를 기준으로 고른다. 안 주면 마지막 패널이 기준이다.
    모델 출력을 기준으로 삼으면 모델이 바뀔 때마다 보는 곳이 달라지므로,
    여러 모델을 비교할 때는 장면 자체(bicubic 등)를 기준으로 넘기는 것이 좋다.
    """
    from matplotlib.patches import Rectangle

    ref = panels[-1][1] if ref is None else ref
    # 큰 씬(인천 1800px)에서 110px 창은 점처럼 보인다. 짧은 변의 1/8 이상은 되게 한다.
    size = max(size, min(ref.shape[:2]) // 8)
    e = cv2.Canny(cv2.cvtColor(ref, cv2.COLOR_RGB2GRAY), 50, 150)
    best, bs = (0, 0), -1.0
    for y in range(0, ref.shape[0] - size, size // 2):
        for x in range(0, ref.shape[1] - size, size // 2):
            v = float(e[y:y + size, x:x + size].mean())
            if v > bs:
                best, bs = (y, x), v
    y, x = best
    H = ref.shape[0]

    n = len(panels)
    fig, ax = plt.subplots(2, n, figsize=(2.9 * n, 6.4), squeeze=False)
    for j, (name, im) in enumerate(panels):
        # 패널마다 해상도가 다를 수 있으니 비율로 환산한다
        f = im.shape[0] / H
        yy, xx, ss = int(y * f), int(x * f), max(4, int(size * f))

        a = ax[0][j]
        a.imshow(im, interpolation='nearest')
        a.add_patch(Rectangle((xx, yy), ss, ss, fill=False, ec='#ffcc00', lw=1.6))
        a.set_title(name, fontsize=9)
        a.set_xticks([]); a.set_yticks([])

        b = ax[1][j]
        b.imshow(im[yy:yy + ss, xx:xx + ss], interpolation='nearest')
        b.set_xticks([]); b.set_yticks([])

    ax[0][0].set_ylabel('전체', fontsize=10)
    ax[1][0].set_ylabel(f'확대 ({size}px)', fontsize=10)
    if title:
        fig.suptitle(title, fontsize=10)
    plt.tight_layout(); plt.show()


def retarget(hr, lr, scale):
    """HR 을 LR x scale 크기로 줄인다. 입력 LR 은 절대 건드리지 않는다."""
    if scale == 3:
        return hr
    h, w = lr.shape[:2]
    return cv2.resize(hr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


def compare(upscale_fn, split='validation', plot=True, label='model', scale=3):
    """검증셋 전체를 bicubic 과 비교해 표와 그래프를 낸다."""
    rows = []
    for stem in list_split(split):
        lr, hr = pair(split, stem)
        hr = retarget(hr, lr, scale)
        pb, sb = score(bicubic(lr, scale), hr)
        pm, sm = score(upscale_fn(lr), hr)
        rows.append((stem.rsplit('_y', 1)[0].replace('AOI_', ''), pb, sb, pm, sm))

    m = np.array([[r[1], r[2], r[3], r[4]] for r in rows]).mean(0)
    print(f'{"":12s}{"PSNR":>10s}{"SSIM":>10s}')
    print(f'{"Bicubic":12s}{m[0]:10.2f}{m[1]:10.4f}')
    print(f'{label:12s}{m[2]:10.2f}{m[3]:10.4f}')
    print(f'{"차이":12s}{m[2] - m[0]:+10.2f}{m[3] - m[1]:+10.4f}   ({len(rows)}장 평균)')
    if not plot:
        return rows

    seen, labels = {}, []
    for n, *_ in rows:
        seen[n] = seen.get(n, 0) + 1
        labels.append(n if sum(1 for r in rows if r[0] == n) == 1 else f'{n}-{seen[n]}')
    idx, w = np.arange(len(rows)), 0.38
    fig, ax = plt.subplots(1, 2, figsize=(13, 4))
    for a, (j, k, name) in zip(ax, [(1, 3, 'PSNR (dB)'), (2, 4, 'SSIM')]):
        b, e = [r[j] for r in rows], [r[k] for r in rows]
        a.bar(idx - w / 2, b, w, label='Bicubic', color='#9aa5b1')
        a.bar(idx + w / 2, e, w, label=label, color='#2f6f9f')
        a.set_xticks(idx); a.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
        a.set_title(name); a.set_ylim(min(b + e) * .97, max(b + e) * 1.02)
        a.grid(axis='y', alpha=.3); a.legend(fontsize=8)
    plt.tight_layout(); plt.show()
    return rows


# ──────────────────────────────────────────────────────────────────────
# 실습에서 쓰는 표시 헬퍼. validation 2패치, test 2구역을 한 번에 그린다.
# ──────────────────────────────────────────────────────────────────────

def show_data():
    """1. 데이터 — validation 2패치 + test 2구역을 한눈에."""
    panels = [(f'validation {i+1} ({s.split("_")[1]})', *pair('validation', s))
              for i, s in enumerate(REPS['validation'])]
    panels += [(f'test {i+1} (Incheon)', load_test(i), None) for i in range(len(TESTS))]
    show(panels)


def show_results(upscale_fn, label='model', scale=3):
    """4. 결과 — validation 2패치 각각을 '전체 + 확대' 로."""
    for i, stem in enumerate(REPS['validation']):
        lr, hr = pair('validation', stem)
        hr = retarget(hr, lr, scale)
        zoom([('Original LR', nearest(lr, scale)), ('Bicubic', bicubic(lr, scale)),
              (label, upscale_fn(lr)), ('Target HR', hr)],
             title=f'validation {i + 1} — {stem}')


def show_test(upscale_fn, label='model', scale=3, save=True):
    """6. 최종 테스트 — 인천 2구역 각각을 '전체 + 확대' 로. 정답이 없어 점수는 없다."""
    out = []
    for i in range(len(TESTS)):
        lr = load_test(i)
        bic = bicubic(lr, scale)
        sr = upscale_fn(lr)
        zoom([('Original LR', nearest(lr, scale)), ('Bicubic', bic), (label, sr)],
             ref=bic, title=f'test {i + 1} (Incheon) — 정답 없음')
        if save:
            f = f'incheon{i + 1}_{label.lower()}.png'
            imageio.imwrite(f, sr)
            print(f'{f} 저장')
        out.append(sr)
    return out
