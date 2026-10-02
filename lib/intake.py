"""Data intake check: read the header, check the statistics, look at a thumbnail."""
import matplotlib.pyplot as plt
import numpy as np
import rasterio

__all__ = ['info', 'stats', 'preview', 'np', 'plt', 'rasterio']


def info(path):
    """Step 1 - what the header says."""
    with rasterio.open(path) as d:
        print(f'size        {d.width} x {d.height}')
        print(f'bands       {d.count}')
        print(f'dtype       {d.dtypes[0]}')
        print(f'CRS         {d.crs}')
        print(f'pixel size  {abs(d.res[0]):.3f} x {abs(d.res[1]):.3f} m')
        print(f'nodata      {d.nodata}')


def stats(path, bits=12):
    """Step 2 - do the numbers fit the bit depth?"""
    ceiling = 2 ** bits - 1
    with rasterio.open(path) as d:
        print(f'{"band":>5} {"min":>7} {"max":>7} {"mean":>8} {"std":>7} '
              f'{"px at ceiling":>14}')
        for b in range(1, d.count + 1):
            v = d.read(b).astype(np.float32)
            print(f'{b:5d} {v.min():7.0f} {v.max():7.0f} {v.mean():8.1f} {v.std():7.1f}'
                  f' {int((v >= ceiling).sum()):14d}')
    print(f'\n{bits}-bit range is 0 - {ceiling}')


def preview(path, size=800, lo=2, hi=98, bits=12):
    """Step 3 - look at it.  The flag map is built at full resolution, then reduced
    with `any`, so single odd pixels do not get averaged away."""
    ceiling = 2 ** bits - 1
    with rasterio.open(path) as d:
        a = d.read((1, 2, 3)).astype(np.float32)
    k = max(1, min(a.shape[1], a.shape[2]) // size)
    h, w = (a.shape[1] // k) * k, (a.shape[2] // k) * k
    a = a[:, :h, :w]

    thumb = a.reshape(3, h // k, k, w // k, k).mean(axis=(2, 4))
    q = np.percentile(thumb, (lo, hi), axis=(1, 2))
    img = np.clip((thumb - q[0][:, None, None]) /
                  (q[1] - q[0])[:, None, None], 0, 1).transpose(1, 2, 0)

    def mark(mask, grow=3):
        m = mask.reshape(h // k, k, w // k, k).any(axis=(1, 3))
        for _ in range(grow):                 # 몇 개 안 되는 화소를 눈에 보이게 키운다
            m |= np.roll(m, 1, 0) | np.roll(m, -1, 0) | np.roll(m, 1, 1) | np.roll(m, -1, 1)
        return m

    n_red = int((a >= ceiling).any(0).sum())
    n_blue = int((a <= 0).any(0).sum())
    flag = img.copy()
    flag[mark((a >= ceiling).any(0))] = (1, 0, 0)
    flag[mark((a <= 0).any(0))] = (0, 0.4, 1)

    fig, axes = plt.subplots(1, 2, figsize=(13, 6.6), layout='constrained')
    axes[0].imshow(img)
    axes[1].imshow(flag)
    for ax, t in zip(axes, ('1.  thumbnail',
                            f'2.  red = {ceiling}  ({n_red} px),   '
                            f'blue = 0  ({n_blue} px)')):
        ax.set_xticks([]); ax.set_yticks([]); ax.set_box_aspect(1)
        ax.set_title(t, fontsize=18, pad=10, loc='left')
    plt.show()
