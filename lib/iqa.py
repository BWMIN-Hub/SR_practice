"""화질 지표 모음 — 참조가 있는 것 셋, 없는 것 셋.

  FR-IQA (Full Reference, IKONOS 영상이 있어야 한다)
    psnr    화소를 하나씩 맞대 본 오차의 평균. 높을수록 좋다
    ssim    화소 주변 창을 통째로 견준다. 1 에 가까울수록 좋다
    ergas   밴드별 오차를 밴드 밝기로 나눠 모은다. 0 에 가까울수록 좋다

  NR-IQA (No Reference, 결과 한 장만 있으면 된다)
    niqe     자연 영상다움에서 얼마나 벗어났나. 낮을수록 좋다  (pyiqa)
    brisque  분포 모양 36개를 학습된 회귀에 넣어 점수. 낮을수록 좋다  (pyiqa)
    piqe     16x16 조각마다 이음매·잡음을 재서 평균. 낮을수록 좋다

NR 지표는 sr_data/metrics.py 와 같은 구현이다. 입력은 비트 깊이에 맞춰 uint8 로
바꾼 뒤 잰다 (uint16 은 4095 / 16383 / 65535 중 맞는 최대값으로 나눈다).
NIQE·BRISQUE 는 pyiqa 가 필요하다:  pip install pyiqa
"""
import cv2
import numpy as np

__all__ = ['psnr', 'ssim', 'ergas', 'niqe', 'brisque', 'piqe',
           'FR', 'NR', 'ALL', 'BETTER', 'evaluate']

SHAVE = 4

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


def _to_uint8(image):
    """
    Convert image to uint8 [0, 255] using bit-depth aware scaling.
    Matches evaluate_sr_improvement.py to_uint8_0_255().

    Args:
        image (numpy.ndarray): Image of any dtype (uint8/uint16/float)

    Returns:
        numpy.ndarray: uint8 image [0, 255]
    """
    if image.dtype == np.uint8:
        return image

    img_f = image.astype(np.float64)
    if img_f.size == 0:
        return np.zeros_like(image, dtype=np.uint8)

    img_max = float(np.nanmax(img_f))

    if img_max <= 0:
        return np.zeros_like(image, dtype=np.uint8)

    if img_max <= 1.0:
        scaled = img_f * 255.0
    else:
        if img_max <= 255.0:
            in_max = 255.0
        elif img_max <= 4095.0:
            in_max = 4095.0
        elif img_max <= 16383.0:
            in_max = 16383.0
        elif img_max <= 65535.0:
            in_max = 65535.0
        else:
            in_max = img_max
        scaled = (np.clip(img_f, 0.0, in_max) / in_max) * 255.0

    return np.clip(np.rint(scaled), 0, 255).astype(np.uint8)


def _prepare_image_tensor(image):
    """
    Convert numpy image to torch tensor for pyiqa metrics.
    First converts to uint8 (bit-depth aware), then normalizes to [0, 1].

    Args:
        image (numpy.ndarray): Image (H x W x C) or (C x H x W), uint8/uint16/float

    Returns:
        torch.Tensor: (1, 3, H, W) float32 tensor normalized to [0, 1]
    """
    import torch

    # Convert to uint8 first (handles uint16, float, etc.)
    image = _to_uint8(image)

    # (H, W, C) -> (C, H, W)
    if len(image.shape) == 3 and image.shape[2] in [1, 3, 4]:
        image = image.transpose(2, 0, 1)
    elif len(image.shape) == 2:
        image = image[np.newaxis, :, :]  # (H, W) -> (1, H, W)

    image_normalized = image.astype(np.float32) / 255.0
    image_tensor = torch.from_numpy(image_normalized).unsqueeze(0)

    # 3채널(RGB)로 변환
    if image_tensor.shape[1] == 4:
        image_tensor = image_tensor[:, :3, :, :]
    elif image_tensor.shape[1] == 1:
        image_tensor = image_tensor.repeat(1, 3, 1, 1)

    return image_tensor


def _prepare_grayscale_for_piqe(image):
    """Convert numpy image to grayscale float64 array scaled to [0, 255]."""
    if len(image.shape) == 3 and image.shape[0] in [1, 3, 4] and image.shape[0] < image.shape[1]:
        image = image.transpose(1, 2, 0)

    if len(image.shape) == 3:
        if image.shape[2] >= 3:
            gray = cv2.cvtColor(image[:, :, :3], cv2.COLOR_RGB2GRAY)
        else:
            gray = image[:, :, 0]
    else:
        gray = image.copy()

    gray = gray.astype(np.float64)
    max_val = gray.max()
    if max_val > 0:
        gray = np.round(255.0 * (gray / max_val))
    return gray


def _normalize_img_with_gauss(image, kernel_size=7, sigma=7.0 / 6.0):
    """Compute MSCN coefficients using Gaussian local normalization."""
    mu = cv2.GaussianBlur(
        image, (kernel_size, kernel_size), sigmaX=sigma, borderType=cv2.BORDER_REPLICATE
    )
    sigma_map = np.sqrt(
        np.abs(
            cv2.GaussianBlur(
                image * image, (kernel_size, kernel_size), sigmaX=sigma, borderType=cv2.BORDER_REPLICATE
            )
            - mu * mu
        )
    )
    return (image - mu) / (sigma_map + 1.0)


def _piqe_notice_dist_criterion(blocks, window_size, block_impaired_threshold, block_size):
    """Detect blocks with noticeable edge artifacts."""
    top_edge = blocks[:, 0, :]
    seg_top = np.lib.stride_tricks.sliding_window_view(top_edge, window_shape=window_size, axis=1)

    right_edge = blocks[:, block_size - 1]
    seg_right = np.lib.stride_tricks.sliding_window_view(right_edge, window_shape=window_size, axis=1)

    bottom_edge = blocks[:, block_size - 1, :]
    seg_bottom = np.lib.stride_tricks.sliding_window_view(bottom_edge, window_shape=window_size, axis=1)

    left_edge = blocks[:, 0]
    seg_left = np.lib.stride_tricks.sliding_window_view(left_edge, window_shape=window_size, axis=1)

    std_top = np.std(seg_top, axis=-1, ddof=1)
    std_right = np.std(seg_right, axis=-1, ddof=1)
    std_bottom = np.std(seg_bottom, axis=-1, ddof=1)
    std_left = np.std(seg_left, axis=-1, ddof=1)

    return (
        (std_top < block_impaired_threshold).any(axis=1)
        | (std_right < block_impaired_threshold).any(axis=1)
        | (std_bottom < block_impaired_threshold).any(axis=1)
        | (std_left < block_impaired_threshold).any(axis=1)
    )


def _piqe_noise_criterion(blocks, block_size, block_var):
    """Analyze blocks for Gaussian noise distortions."""
    block_sigma = np.sqrt(block_var)
    center1 = (block_size + 1) // 2
    center2 = center1 + 1

    center_cols = np.stack((blocks[:, :, center1 - 1], blocks[:, :, center2 - 1]), axis=-1)
    surround = np.delete(blocks, [center1 - 1, center2 - 1], axis=2)

    center_std = np.std(center_cols, axis=(1, 2), ddof=1)
    surround_std = np.std(surround, axis=(1, 2), ddof=1)

    cen_sur_dev = np.zeros_like(center_std, dtype=np.float64)
    valid = surround_std != 0
    cen_sur_dev[valid] = center_std[valid] / surround_std[valid]

    block_beta = np.abs(block_sigma - cen_sur_dev) / np.maximum(np.maximum(block_sigma, cen_sur_dev), 1e-10)
    return block_sigma, block_beta



# ── 참조가 없는 지표 (sr_data/metrics.py) ───────────────────────────
_PYIQA = {}


def _pyiqa(name):
    if name not in _PYIQA:
        import pyiqa
        _PYIQA[name] = pyiqa.create_metric(name, device='cpu')
    return _PYIQA[name]


def niqe(img):
    """NIQE (pyiqa). 낮을수록 좋다."""
    return float(_pyiqa('niqe')(_prepare_image_tensor(img)).item())


def brisque(img):
    """BRISQUE (pyiqa). 낮을수록 좋다."""
    return float(_pyiqa('brisque')(_prepare_image_tensor(img)).item())


def piqe(image):
    """PIQE. 16x16 조각의 MSCN 으로 이음매·잡음을 잰다. 0 - 100, 낮을수록 좋다."""
    block_size = 16
    activity_threshold = 0.1
    block_impaired_threshold = 0.1
    window_size = 6

    gray = _prepare_grayscale_for_piqe(image)
    rows, cols = gray.shape[:2]

    rows_pad = (block_size - rows % block_size) % block_size
    cols_pad = (block_size - cols % block_size) % block_size
    if rows_pad > 0 or cols_pad > 0:
        gray = np.pad(gray, ((0, rows_pad), (0, cols_pad)), mode='symmetric')

    img_normalized = _normalize_img_with_gauss(gray)

    n_blocks_h = img_normalized.shape[0] // block_size
    n_blocks_w = img_normalized.shape[1] // block_size
    blocks = img_normalized.reshape(n_blocks_h, block_size, n_blocks_w, block_size)
    blocks = blocks.transpose(0, 2, 1, 3).reshape(-1, block_size, block_size)

    block_var = np.var(blocks, axis=(1, 2), ddof=1)
    active_blocks = block_var > activity_threshold
    n_hsa = int(np.sum(active_blocks))
    if n_hsa == 0:
        return 100.0

    block_impaired = _piqe_notice_dist_criterion(
        blocks, window_size, block_impaired_threshold, block_size
    )
    block_sigma, block_beta = _piqe_noise_criterion(blocks, block_size - 1, block_var)
    noise_mask = block_sigma > 2 * block_beta

    dist_block_scores = (
        active_blocks.astype(np.float64) * block_impaired.astype(np.float64) * (1.0 - block_var)
        + active_blocks.astype(np.float64) * noise_mask.astype(np.float64) * block_var
    )
    piqe_score = ((np.sum(dist_block_scores) + 1.0) / (1.0 + n_hsa)) * 100.0
    return float(np.clip(piqe_score, 0.0, 100.0))


def evaluate(sr, hr=None, scale=3):
    """한 장에 대해 잴 수 있는 지표를 모두 낸다. hr 이 없으면 NR 만."""
    out = {}
    if hr is not None:
        out['psnr'] = psnr(sr, hr)
        out['ssim'] = ssim(sr, hr)
        out['ergas'] = ergas(sr, hr, scale)
    out['niqe'] = niqe(sr)
    out['brisque'] = brisque(sr)
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


__all__ += ['report']
