"""Data intake check — read the header, check the statistics, look at the image."""
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.patches import Rectangle
from rasterio.warp import transform as warp_transform

__all__ = ['info', 'show_size', 'show_bands', 'show_dtype', 'show_crs', 'show_pixel',
           'stats', 'preview', 'np', 'plt', 'rasterio']

C_LINE, C_FILL, C_MARK = '#2b6cb0', '#dce6f2', '#c0392b'
BAND_HINT = {4: ['B1  red', 'B2  green', 'B3  blue', 'B4  near infrared'],
             3: ['B1  red', 'B2  green', 'B3  blue']}


def _names(n):
    return BAND_HINT.get(n, [f'B{i}' for i in range(1, n + 1)])


def _read(path, bands=None, size=900):
    """Decimated read — enough for a picture, cheap enough for a notebook."""
    with rasterio.open(path) as d:
        bands = bands or tuple(range(1, d.count + 1))
        k = max(1, min(d.width, d.height) // size)
        a = d.read(bands, out_shape=(len(bands), d.height // k, d.width // k))
    return a.astype(np.float32)


def _st(a, lo=2, hi=98):
    q = np.percentile(a, (lo, hi), axis=(-2, -1), keepdims=True)
    return np.clip((a - q[0]) / (q[1] - q[0] + 1e-9), 0, 1)


def _m(v):
    return f'{v:,.0f} m' if v < 1000 else f'{v:,.0f} m   ({v / 1000:.2f} km)'


# --------------------------------------------------------------- the header
def info(path):
    """Everything the loader needs, before opening a single pixel."""
    with rasterio.open(path) as d:
        print(f'size        {d.width} x {d.height} px')
        print(f'bands       {d.count}')
        print(f'dtype       {d.dtypes[0]}')
        print(f'CRS         {d.crs}')
        print(f'pixel size  {abs(d.res[0]):.3f} x {abs(d.res[1]):.3f} m')
        print(f'nodata      {d.nodata}')


def show_size(path):
    """How big is it — in pixels, and on the ground."""
    with rasterio.open(path) as d:
        w, h, px, py = d.width, d.height, abs(d.res[0]), abs(d.res[1])
    W, H = w * px, h * py
    fig, ax = plt.subplots(figsize=(8.4, 8.4 * h / w), layout='constrained')
    ax.add_patch(Rectangle((0, 0), w, h, fc=C_FILL, ec=C_LINE, lw=2.5))
    ax.annotate('', xy=(0, h * 1.045), xytext=(w, h * 1.045),
                arrowprops=dict(arrowstyle='<->', color=C_LINE, lw=1.8))
    ax.text(w / 2, h * 1.06, f'{w:,} px', ha='center', va='bottom',
            fontsize=19, color=C_LINE)
    ax.text(w / 2, -h * 0.045, _m(W), ha='center', va='top', fontsize=19)
    ax.annotate('', xy=(-w * 0.045, 0), xytext=(-w * 0.045, h),
                arrowprops=dict(arrowstyle='<->', color=C_LINE, lw=1.8))
    ax.text(-w * 0.06, h / 2, f'{h:,} px', ha='right', va='center',
            rotation=90, fontsize=19, color=C_LINE)
    ax.text(w * 1.045, h / 2, _m(H), ha='left', va='center',
            rotation=270, fontsize=19)
    ax.text(w / 2, h / 2, f'{w * h / 1e6:.1f} M pixels\n{W * H / 1e6:.2f} km²',
            ha='center', va='center', fontsize=22, linespacing=1.7)
    bar = 10 ** int(np.log10(W / 4))
    ax.add_patch(Rectangle((w * 0.04, h * 0.05), bar / px, h * 0.012,
                           fc='#333', ec='none'))
    ax.text(w * 0.04 + bar / px / 2, h * 0.075, f'{bar:,.0f} m', ha='center',
            va='bottom', fontsize=15)
    ax.set_xlim(-w * 0.16, w * 1.16); ax.set_ylim(-h * 0.1, h * 1.12)
    ax.set_aspect('equal'); ax.axis('off')
    plt.show()


def show_bands(path, size=520):
    """Each band on its own, then the three visible ones together."""
    a = _read(path, size=size)
    n = len(a)
    names = _names(n)
    fig, axes = plt.subplots(1, n + 1, figsize=(3.4 * (n + 1), 4.2),
                             layout='constrained')
    for i, ax in enumerate(axes[:n]):
        ax.imshow(_st(a[i]), cmap='gray')
        ax.set_title(names[i], fontsize=18, pad=8)
    axes[n].imshow(_st(a[:3]).transpose(1, 2, 0))
    axes[n].set_title('B1 B2 B3  as RGB', fontsize=18, pad=8)
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([]); ax.set_box_aspect(1)
    plt.show()


def show_dtype(path, bits=12):
    """uint16 is the container.  The data inside is narrower than that."""
    with rasterio.open(path) as d:
        dt = d.dtypes[0]
        a = d.read().astype(np.float32)
    top = float(np.iinfo(dt).max)
    lo, hi = float(a.min()), float(a.max())
    ceil = 2 ** bits - 1
    names = _names(len(a))

    fig, axes = plt.subplots(1, 2, figsize=(15.5, 5.4), layout='constrained',
                             gridspec_kw=dict(width_ratios=[1, 1.25]))
    ax = axes[0]
    ax.add_patch(Rectangle((0, 0.3), top, 0.4, fc='#e9edf2', ec='#99a3af', lw=1.4))
    ax.add_patch(Rectangle((lo, 0.3), hi - lo, 0.4, fc=C_LINE, ec='none'))
    ax.axvline(ceil, color=C_MARK, ls='--', lw=2.0)
    ax.text(ceil + top * 0.015, 0.95, f'{bits}-bit   {ceil}', color=C_MARK,
            fontsize=16, ha='left', va='center')
    ax.text(top, 0.95, f'{dt}   {top:,.0f}', fontsize=16, ha='right', va='center')
    ax.text(0, 0.2, f'data   {lo:,.0f} - {hi:,.0f}', color=C_LINE, fontsize=16,
            ha='left', va='center')
    ax.set_xlim(-top * 0.03, top * 1.05); ax.set_ylim(0.05, 1.1)
    ax.set_yticks([]); ax.tick_params(labelsize=14)
    ax.set_xlabel('pixel value', fontsize=17)
    ax.set_title('1.  the container', fontsize=19, loc='left', pad=10)
    for sp in ('top', 'right', 'left'):
        ax.spines[sp].set_visible(False)

    ax = axes[1]
    bins = np.linspace(0, ceil, 300)
    for i, v in enumerate(a):
        ax.hist(v.ravel(), bins=bins, histtype='step', lw=2.2,
                weights=np.full(v.size, 1.0 / v.size),
                label=f'{names[i]}    mean {v.mean():.0f}   std {v.std():.0f}')
    ax.axvline(ceil, color=C_MARK, ls='--', lw=2.0)
    ax.set_yscale('log'); ax.set_yticks([]); ax.set_xlim(0, ceil)
    ax.tick_params(labelsize=14)
    ax.set_xlabel('pixel value', fontsize=17)
    ax.legend(fontsize=14, frameon=False, loc='upper right')
    ax.set_title('2.  what is actually in it', fontsize=19, loc='left', pad=10)
    for sp in ('top', 'right', 'left'):
        ax.spines[sp].set_visible(False)
    plt.show()


def show_crs(path):
    """Which coordinate system the pixels are pinned to."""
    with rasterio.open(path) as d:
        crs, b = d.crs, d.bounds
    xs = [b.left, b.right, b.right, b.left]
    ys = [b.top, b.top, b.bottom, b.bottom]
    lon, lat = warp_transform(crs, 'EPSG:4326', xs, ys)

    fig, ax = plt.subplots(figsize=(9.6, 7.6), layout='constrained')
    ax.add_patch(Rectangle((b.left, b.bottom), b.right - b.left, b.top - b.bottom,
                           fc=C_FILL, ec=C_LINE, lw=2.5))
    off = [(-1, 1), (1, 1), (1, -1), (-1, -1)]
    pad = (b.right - b.left) * 0.02
    for (x, y, lo_, la, (ox, oy)) in zip(xs, ys, lon, lat, off):
        ax.plot(x, y, 'o', color=C_MARK, ms=8, zorder=3)
        ax.text(x + ox * pad, y + oy * pad,
                f'E {x:,.0f}   N {y:,.0f}\n{la:.4f}°N  {lo_:.4f}°E',
                fontsize=14, linespacing=1.5,
                ha='left' if ox > 0 else 'right',
                va='bottom' if oy > 0 else 'top')
    name = crs.to_wkt().split('"')[1] if crs else '-'
    ax.text(0.5, 0.5, f'{crs}\n{name}\nunits: {crs.linear_units}',
            transform=ax.transAxes, ha='center', va='center',
            fontsize=20, linespacing=1.9)
    m = (b.right - b.left) * 0.34
    ax.set_xlim(b.left - m, b.right + m); ax.set_ylim(b.bottom - m, b.top + m)
    ax.set_aspect('equal'); ax.axis('off')
    plt.show()


def show_pixel(path, col=None, row=None, steps=(240, 24)):
    """Zoom until one pixel fills the panel, and say how wide it is on the ground."""
    with rasterio.open(path) as d:
        W, H, px, py = d.width, d.height, abs(d.res[0]), abs(d.res[1])
        col = W // 2 if col is None else col
        row = H // 2 if row is None else row
        wins, imgs = [], [_st(_read(path, (1, 2, 3), size=600)).transpose(1, 2, 0)]
        for s in steps:
            c0, r0 = int(np.clip(col - s // 2, 0, W - s)), int(np.clip(row - s // 2, 0, H - s))
            wins.append((c0, r0, s))
            imgs.append(_st(d.read((1, 2, 3),
                                   window=rasterio.windows.Window(c0, r0, s, s))
                            .astype(np.float32)).transpose(1, 2, 0))
        one = d.read(window=rasterio.windows.Window(col, row, 1, 1))[:, 0, 0]
    # 마지막 확대창과 같은 창으로 늘려야 색이 화면과 맞는다
    c0, r0, _ = wins[-1]
    rgb = imgs[-1][row - r0, col - c0]

    fig, axes = plt.subplots(1, len(imgs) + 1, figsize=(4.0 * (len(imgs) + 1), 4.8),
                             layout='constrained')
    scales = [(W, H)] + [(s, s) for _, _, s in wins]
    prev = (0, 0)
    for i, ax in enumerate(axes[:-1]):
        ax.imshow(imgs[i], interpolation='nearest')
        sw, sh = scales[i]
        k = imgs[i].shape[1] / sw
        if i < len(wins):
            c0, r0, s = wins[i]
            ax.add_patch(Rectangle(((c0 - prev[0]) * k, (r0 - prev[1]) * k),
                                   s * k, s * k, fill=False, ec=C_MARK, lw=2.6))
            prev = (c0, r0)
        ax.set_title(f'{sw} x {sh} px', fontsize=18, pad=8)
        ax.set_xlabel(f'{sw * px:,.0f} x {sh * py:,.0f} m', fontsize=16)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_box_aspect(1)

    ax = axes[-1]
    ax.add_patch(Rectangle((0, 0), 1, 1, fc=rgb, ec=C_MARK, lw=3.0))
    ax.annotate('', xy=(0, -0.08), xytext=(1, -0.08),
                arrowprops=dict(arrowstyle='<->', color=C_MARK, lw=1.8))
    ax.text(0.5, -0.13, f'{px:.2f} m', ha='center', va='top', fontsize=19,
            color=C_MARK)
    ax.annotate('', xy=(-0.08, 0), xytext=(-0.08, 1),
                arrowprops=dict(arrowstyle='<->', color=C_MARK, lw=1.8))
    ax.text(-0.13, 0.5, f'{py:.2f} m', ha='right', va='center', rotation=90,
            fontsize=19, color=C_MARK)
    ax.text(0.5, 1.06, '1 x 1 px', ha='center', va='bottom', fontsize=18)
    ax.text(0.5, 0.5, '  '.join(str(int(v)) for v in one), ha='center',
            va='center', fontsize=17,
            color='white' if rgb.mean() < 0.5 else 'black')
    ax.set_xlim(-0.3, 1.3); ax.set_ylim(-0.3, 1.3)
    ax.set_aspect('equal'); ax.axis('off')
    plt.show()


# --------------------------------------------------------------- 2 and 3
def stats(path, bits=12):
    """Do min / max / mean / std fit the bit depth?"""
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
    """The flag map is built at full resolution, then reduced with `any`, so single
    odd pixels do not get averaged away."""
    ceiling = 2 ** bits - 1
    with rasterio.open(path) as d:
        a = d.read((1, 2, 3)).astype(np.float32)
    k = max(1, min(a.shape[1], a.shape[2]) // size)
    h, w = (a.shape[1] // k) * k, (a.shape[2] // k) * k
    a = a[:, :h, :w]

    thumb = a.reshape(3, h // k, k, w // k, k).mean(axis=(2, 4))
    img = _st(thumb, lo, hi).transpose(1, 2, 0)

    def mark(mask, grow=3):
        m = mask.reshape(h // k, k, w // k, k).any(axis=(1, 3))
        for _ in range(grow):
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
