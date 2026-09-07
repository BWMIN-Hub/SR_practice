"""화질 지표 모음 — 참조가 있는 것 셋, 없는 것 셋.

  FR-IQA (Full Reference, 정답이 있어야 한다)
    psnr    화소를 하나씩 맞대 본 오차의 평균. 높을수록 좋다
    ssim    화소 주변 창을 통째로 견준다. 1 에 가까울수록 좋다
    ergas   밴드별 오차를 밴드 밝기로 나눠 모은다. 0 에 가까울수록 좋다

  NR-IQA (No Reference, 결과 한 장만 있으면 된다)
    niqe     자연 영상다움에서 얼마나 벗어났나. 낮을수록 좋다
    brisque  분포 모양 36개를 학습된 회귀에 넣어 점수. 낮을수록 좋다
    piqe     16x16 조각마다 규칙으로 채점해 평균. 낮을수록 좋다

주의 — NIQE 와 BRISQUE 는 "정상 영상이란 이런 것" 이라는 기준 모델이 필요하다.
원 논문은 자연 사진 수백 장(LIVE)으로 그 기준을 만들었는데, 위성 영상은 통계가
달라 그대로 쓰면 뜻이 없다. 그래서 이 프로젝트의 IKONOS HR 로 기준을 다시 잡았다.
  -> results/iqa/niqe_model.npz, brisque_svr.npz  (tools/fit_iqa.py 가 만든다)
따라서 값은 **이 데이터 안에서의 상대 비교**로만 읽어야 하고, 다른 논문의
NIQE/BRISQUE 수치와 직접 견주면 안 된다. PIQE 는 기준 모델이 없어 그대로 쓴다.
"""
import os

import cv2
import numpy as np
from scipy.special import gamma as _G

__all__ = ['psnr', 'ssim', 'ergas', 'niqe', 'brisque', 'piqe',
           'FR', 'NR', 'ALL', 'BETTER', 'evaluate', 'load_models']

SHAVE = 4
_ALPHA = np.arange(0.2, 10.001, 0.001)
_RHO = _G(1 / _ALPHA) * _G(3 / _ALPHA) / _G(2 / _ALPHA) ** 2
_RA = _G(2 / _ALPHA) ** 2 / (_G(1 / _ALPHA) * _G(3 / _ALPHA))
_DIRS = [(0, 1), (1, 0), (1, 1), (1, -1)]

FR = ['psnr', 'ssim', 'ergas']
NR = ['niqe', 'brisque', 'piqe']
ALL = FR + NR
BETTER = {'psnr': 'high', 'ssim': 'high', 'ergas': 'low',
          'niqe': 'low', 'brisque': 'low', 'piqe': 'low'}


# ── 참조가 있는 지표 ──────────────────────────────────────────────────
def _shave(x):
    return x[SHAVE:-SHAVE, SHAVE:-SHAVE]


def psnr(sr, hr):
    from skimage.metrics import peak_signal_noise_ratio
    return float(peak_signal_noise_ratio(_shave(hr), _shave(sr), data_range=255))


def ssim(sr, hr):
    from skimage.metrics import structural_similarity
    return float(structural_similarity(_shave(hr), _shave(sr),
                                       data_range=255, channel_axis=2))


def ergas(sr, hr, scale=3):
    """h/l = 1/scale. 밴드별 RMSE 를 그 밴드의 평균 밝기로 나눠 모은다."""
    a, b = _shave(sr).astype(np.float64), _shave(hr).astype(np.float64)
    rmse = np.sqrt(((a - b) ** 2).reshape(-1, a.shape[-1]).mean(0))
    mu = b.reshape(-1, b.shape[-1]).mean(0)
    return float(100 / scale * np.sqrt(((rmse / mu) ** 2).mean()))


# ── 공통 도구 (NIQE·BRISQUE·PIQE 가 모두 MSCN 에서 시작한다) ──────────
def mscn(gray, ksize=7, sigma=7 / 6):
    """국소 평균을 빼고 국소 표준편차로 나눈다. 밝기·대비가 사라지고 질감만 남는다."""
    x = gray.astype(np.float64)
    mu = cv2.GaussianBlur(x, (ksize, ksize), sigma)
    sq = cv2.GaussianBlur(x * x, (ksize, ksize), sigma)
    sd = np.sqrt(np.abs(sq - mu * mu))
    return (x - mu) / (sd + 1.0)


def _gray(img):
    return (cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float64)
            if img.ndim == 3 else img.astype(np.float64))


def _fit_ggd(v):
    s2 = float(np.mean(v ** 2)); m = float(np.mean(np.abs(v)))
    if m == 0:
        return 2.0, s2
    return float(_ALPHA[np.argmin(np.abs(_RHO - s2 / m ** 2))]), s2


def _fit_aggd(v):
    l, r = v[v < 0], v[v > 0]
    sl = np.sqrt(np.mean(l ** 2)) if l.size else 1e-6
    sr_ = np.sqrt(np.mean(r ** 2)) if r.size else 1e-6
    g = sl / sr_
    R = np.mean(np.abs(v)) ** 2 / np.mean(v ** 2)
    Rp = R * (g ** 3 + 1) * (g + 1) / (g ** 2 + 1) ** 2
    a = float(_ALPHA[np.argmin(np.abs(_RA - Rp))])
    return a, float((sr_ - sl) * _G(2 / a) / _G(1 / a)), float(sl ** 2), float(sr_ ** 2)


def _feat18(m):
    """한 스케일에서 GGD 2 + AGGD 4x4 = 18개."""
    f = list(_fit_ggd(m.ravel()))
    for dy, dx in _DIRS:
        f += list(_fit_aggd((m * np.roll(np.roll(m, dy, 0), dx, 1)).ravel()))
    return f


def features36(img):
    """BRISQUE 특징 36개 (원본 + 1/2 축소)."""
    x, out = _gray(img), []
    for _ in range(2):
        out += _feat18(mscn(x))
        x = cv2.resize(x, (x.shape[1] // 2, x.shape[0] // 2),
                       interpolation=cv2.INTER_AREA)
    return np.array(out)


# ── 참조가 없는 지표 ──────────────────────────────────────────────────
_M = {}


def load_models(path=None):
    """기준 모델(NIQE 평균·공분산, BRISQUE 회귀)을 읽는다. 없으면 안내한다."""
    if _M:
        return _M
    root = path or os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', 'results', 'iqa')
    for k, f in (('niqe', 'niqe_model.npz'), ('brisque', 'brisque_svr.npz')):
        p = os.path.join(root, f)
        if not os.path.exists(p):
            raise FileNotFoundError(
                f'{f} 가 없다. tools/fit_iqa.py 로 만들거나 저장소에서 내려받아야 한다.')
        _M[k] = dict(np.load(p, allow_pickle=True))
    return _M


def niqe(img, patch=96, path=None):
    """자연 영상다움에서 벗어난 정도. 기준 분포와의 거리다."""
    M = load_models(path)['niqe']
    g = _gray(img)
    ny, nx = g.shape[0] // patch, g.shape[1] // patch
    feats = []
    for by in range(max(ny, 1)):
        for bx in range(max(nx, 1)):
            b = g[by * patch:(by + 1) * patch, bx * patch:(bx + 1) * patch]
            if min(b.shape) < patch // 2:
                continue
            m = mscn(b)
            if m.var() < 0.05:                 # 밋밋한 조각은 뺀다
                continue
            feats.append(_feat18(m))
    if not feats:
        return float('nan')
    F = np.array(feats)
    mu, cov = F.mean(0), np.cov(F.T) if len(F) > 1 else np.eye(F.shape[1])
    mu0, cov0 = M['mu'], M['cov']
    d = mu0 - mu
    inv = np.linalg.pinv((cov0 + cov) / 2)
    return float(np.sqrt(max(d @ inv @ d, 0)))


def brisque(img, path=None):
    """36개 특징을 학습된 회귀에 넣어 점수 하나."""
    M = load_models(path)['brisque']
    x = (features36(img) - M['mu']) / M['sd']
    k = np.exp(-M['gamma'] * ((M['sv'] - x) ** 2).sum(1))
    return float(np.clip(M['coef'] @ k + M['b'], 0, 100))


def piqe(img, blk=16, act_th=0.1, imp_th=0.1):
    """16x16 조각마다 이음매·잡음을 재서 평균. 기준 모델이 필요 없다."""
    m = mscn(_gray(img))
    ny, nx = m.shape[0] // blk, m.shape[1] // blk
    n_act = 0.0; bad = 0.0
    for by in range(ny):
        for bx in range(nx):
            b = m[by * blk:(by + 1) * blk, bx * blk:(bx + 1) * blk]
            v = float(b.var())
            if v <= act_th:
                continue
            n_act += 1
            gy, gx = np.abs(np.diff(b, axis=0)), np.abs(np.diff(b, axis=1))
            edge = max(gy[0].mean(), gy[-1].mean(), gx[:, 0].mean(), gx[:, -1].mean())
            inner = 0.5 * (gy[1:-1].mean() + gx[:, 1:-1].mean()) + 1e-9
            if edge > inner * (1 + imp_th):
                bad += v
            else:
                res = b - cv2.GaussianBlur(b, (3, 3), 0.8)
                if res.var() > 0.35 * v:
                    bad += v
    return float(np.clip(100 * (bad + 1.0) / (n_act + 1.0), 0, 100))


def evaluate(sr, hr=None, scale=3, path=None):
    """한 장에 대해 잴 수 있는 지표를 모두 낸다. hr 이 없으면 NR 만."""
    out = {}
    if hr is not None:
        out['psnr'] = psnr(sr, hr)
        out['ssim'] = ssim(sr, hr)
        out['ergas'] = ergas(sr, hr, scale)
    out['niqe'] = niqe(sr, path=path)
    out['brisque'] = brisque(sr, path=path)
    out['piqe'] = piqe(sr)
    return out


# ── 보여주기 ─────────────────────────────────────────────────────────
def report(items, metrics=None, crop=None, title='', figsize_w=2.0):
    """영상과 그 점수를 한 줄에 나란히 놓는다.

    items   : [(이름, 영상, {지표: 값}), ...]
    crop    : (x, y, size) 로 확대해 볼 자리. 없으면 가운데
    색은 그 지표 안에서의 순위다 — 초록이 좋고 붉은 쪽이 나쁘다.
    글자는 모두 영문이다. Colab 기본 폰트에 한글이 없어서다.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    ks = metrics or [k for k in ALL if k in items[0][2]]
    cmap = LinearSegmentedColormap.from_list('gr', ['#2e8b57', '#f0e68c', '#c0504d'])
    n = len(items)
    fig, ax = plt.subplots(n, 1 + len(ks), squeeze=False,
                           figsize=(figsize_w * (1 + len(ks)), 1.28 * n),
                           gridspec_kw=dict(width_ratios=[1.5] + [1] * len(ks),
                                            wspace=.06, hspace=.06))
    # 지표마다 순위를 매겨 색을 정한다
    rank = {}
    for k in ks:
        v = [it[2][k] for it in items]
        order = sorted(range(n), key=lambda i: v[i], reverse=(BETTER[k] == 'high'))
        for pos, i in enumerate(order):
            rank[(i, k)] = pos / max(n - 1, 1)

    for i, (name, img, vals) in enumerate(items):
        a = ax[i][0]
        if crop:
            x, y, s = crop
            img = img[y:y + s, x:x + s]
        a.imshow(img, interpolation='nearest')
        a.set_xticks([]); a.set_yticks([])
        a.set_ylabel(name, fontsize=11, rotation=0, ha='right', va='center',
                     labelpad=8)
        for j, k in enumerate(ks, start=1):
            b = ax[i][j]
            b.set_facecolor(cmap(rank[(i, k)]))
            b.text(.5, .5, f'{vals[k]:.3f}' if vals[k] < 10 else f'{vals[k]:.2f}',
                   ha='center', va='center', fontsize=13, color='white',
                   weight='bold', transform=b.transAxes)
            b.set_xticks([]); b.set_yticks([])
            if i == 0:
                arrow = '↑' if BETTER[k] == 'high' else '↓'
                b.set_title(f'{k.upper()} {arrow}', fontsize=12, pad=8)
    ax[0][0].set_title('result' + (f'  (zoom {crop[2]}px)' if crop else ''),
                       fontsize=12, pad=8)
    if title:
        fig.suptitle(title, fontsize=13, y=1.0)
    plt.show()


__all__ += ['report', 'features36', 'mscn']
