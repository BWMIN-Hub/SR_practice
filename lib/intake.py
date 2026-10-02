"""Data intake check — read the header, check the statistics, look at the image.

The file is a GeoTIFF, so everything the header holds sits in TIFF tags: pixel
scale (33550), origin (33922), the projection keys (34735) and their names
(34737).  tifffile reads those directly, which keeps this page free of installs.
In real work you would reach for gdalinfo or rasterio instead.
"""
import matplotlib.pyplot as plt
import numpy as np
import tifffile
from matplotlib.patches import Rectangle

__all__ = ['info', 'show_size', 'show_bands', 'show_dtype', 'show_crs', 'show_pixel',
           'stats', 'preview', 'meta', 'read', 'np', 'plt']

C_LINE, C_FILL, C_MARK = '#2b6cb0', '#dce6f2', '#c0392b'
BAND_HINT = {4: ['B1  red', 'B2  green', 'B3  blue', 'B4  near infrared'],
             3: ['B1  red', 'B2  green', 'B3  blue']}
_CACHE = {}


# --------------------------------------------------------------- reading
def _load(path):
    """Read the file once, keep it, and pull the GeoTIFF tags out of the header."""
    if path not in _CACHE:
        with tifffile.TiffFile(path) as t:
            p = t.pages[0]
            a = p.asarray()
            g = {c: (p.tags[c].value if p.tags.get(c) else None)
                 for c in (33550, 33922, 34735, 34737, 42113)}
        a = a[:, :, None] if a.ndim == 2 else a
        _CACHE[path] = (np.ascontiguousarray(a.transpose(2, 0, 1)), g)
    return _CACHE[path]


def meta(path):
    a, g = _load(path)
    scale = g[33550] or (1.0, 1.0, 0.0)
    tie = g[33922] or (0.0,) * 6
    keys = g[34735] or ()
    kv = {}
    for i in range(4, len(keys) - 3, 4):            # 4개 헤더 뒤로 (키, 위치, 개수, 값)
        kid, loc, _, val = keys[i:i + 4]
        if loc == 0:
            kv[kid] = val
    names = [s for s in (g[34737] or '').split('|') if s]
    left, top = float(tie[3]), float(tie[4])
    h, w = a.shape[1], a.shape[2]
    return dict(count=a.shape[0], height=h, width=w, dtype=str(a.dtype),
                res=(float(scale[0]), float(scale[1])),
                bounds=(left, top - h * scale[1], left + w * scale[0], top),
                epsg=kv.get(3072) or kv.get(2048),
                name=names[0] if names else '-',
                units='metre' if kv.get(3076) == 9001 else '-',
                nodata=float(g[42113]) if g[42113] else None)


def read(path):
    """The pixels, as (band, row, col).  Cached, so repeated calls are free."""
    return _load(path)[0]


def _names(n):
    return BAND_HINT.get(n, [f'B{i}' for i in range(1, n + 1)])


def _st(a, lo=2, hi=98):
    q = np.percentile(a, (lo, hi), axis=(-2, -1), keepdims=True)
    return np.clip((a - q[0]) / (q[1] - q[0] + 1e-9), 0, 1)


def _thumb(path, bands=(0, 1, 2), size=600):
    a, _ = _load(path)
    k = max(1, min(a.shape[1], a.shape[2]) // size)
    return a[list(bands), ::k, ::k].astype(np.float32)


def _m(v):
    return f'{v:,.0f} m' if v < 1000 else f'{v:,.0f} m   ({v / 1000:.2f} km)'


def _utm_to_ll(E, N, zone, north=True):
    """Inverse transverse Mercator on WGS 84 — matches a full PROJ transform to
    well under a millimetre, and needs nothing installed."""
    a, f = 6378137.0, 1 / 298.257223563
    e2 = f * (2 - f); ep2 = e2 / (1 - e2); k0 = 0.9996
    e1 = (1 - np.sqrt(1 - e2)) / (1 + np.sqrt(1 - e2))
    x = np.asarray(E, float) - 500000.0
    y = np.asarray(N, float) - (0.0 if north else 1e7)
    mu = (y / k0) / (a * (1 - e2 / 4 - 3 * e2 ** 2 / 64 - 5 * e2 ** 3 / 256))
    p1 = (mu + (3 * e1 / 2 - 27 * e1 ** 3 / 32) * np.sin(2 * mu)
          + (21 * e1 ** 2 / 16 - 55 * e1 ** 4 / 32) * np.sin(4 * mu)
          + (151 * e1 ** 3 / 96) * np.sin(6 * mu)
          + (1097 * e1 ** 4 / 512) * np.sin(8 * mu))
    C1 = ep2 * np.cos(p1) ** 2; T1 = np.tan(p1) ** 2
    N1 = a / np.sqrt(1 - e2 * np.sin(p1) ** 2)
    R1 = a * (1 - e2) / (1 - e2 * np.sin(p1) ** 2) ** 1.5
    D = x / (N1 * k0)
    lat = p1 - (N1 * np.tan(p1) / R1) * (
        D ** 2 / 2 - (5 + 3 * T1 + 10 * C1 - 4 * C1 ** 2 - 9 * ep2) * D ** 4 / 24
        + (61 + 90 * T1 + 298 * C1 + 45 * T1 ** 2 - 252 * ep2 - 3 * C1 ** 2)
        * D ** 6 / 720)
    lon = np.radians(zone * 6 - 183) + (
        D - (1 + 2 * T1 + C1) * D ** 3 / 6
        + (5 - 2 * C1 + 28 * T1 - 3 * C1 ** 2 + 8 * ep2 + 24 * T1 ** 2)
        * D ** 5 / 120) / np.cos(p1)
    return np.degrees(lon), np.degrees(lat)


# --------------------------------------------------------------- 1. the header
def info(path):
    """Everything the loader needs, before opening a single pixel."""
    m = meta(path)
    print(f'size        {m["width"]} x {m["height"]} px')
    print(f'bands       {m["count"]}')
    print(f'dtype       {m["dtype"]}')
    print(f'CRS         EPSG:{m["epsg"]}   {m["name"]}')
    print(f'pixel size  {m["res"][0]:.3f} x {m["res"][1]:.3f} {m["units"]}')
    print(f'nodata      {m["nodata"]}')


def show_size(path):
    """How big is it — in pixels, and on the ground."""
    m = meta(path)
    w, h = m['width'], m['height']
    px, py = m['res']
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
    a, _ = _load(path)
    n = a.shape[0]
    t = _thumb(path, tuple(range(n)), size)
    names = _names(n)
    fig, axes = plt.subplots(1, n + 1, figsize=(3.4 * (n + 1), 4.2),
                             layout='constrained')
    for i, ax in enumerate(axes[:n]):
        ax.imshow(_st(t[i]), cmap='gray')
        ax.set_title(names[i], fontsize=18, pad=8)
    axes[n].imshow(_st(t[:3]).transpose(1, 2, 0))
    axes[n].set_title('B1 B2 B3  as RGB', fontsize=18, pad=8)
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([]); ax.set_box_aspect(1)
    plt.show()


def show_dtype(path, bits=12):
    """uint16 is the container.  The data inside is narrower than that."""
    a, _ = _load(path)
    a = a.astype(np.float32)
    dt = meta(path)['dtype']
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
    m = meta(path)
    left, bottom, right, top = m['bounds']
    xs = [left, right, right, left]
    ys = [top, top, bottom, bottom]
    epsg = m['epsg'] or 0
    utm = 32600 < epsg < 32761
    if utm:
        zone = epsg % 100
        lon, lat = _utm_to_ll(xs, ys, zone, north=epsg < 32700)

    fig, ax = plt.subplots(figsize=(9.6, 7.6), layout='constrained')
    ax.add_patch(Rectangle((left, bottom), right - left, top - bottom,
                           fc=C_FILL, ec=C_LINE, lw=2.5))
    off = [(-1, 1), (1, 1), (1, -1), (-1, -1)]
    pad = (right - left) * 0.02
    for i, (x, y, (ox, oy)) in enumerate(zip(xs, ys, off)):
        ax.plot(x, y, 'o', color=C_MARK, ms=8, zorder=3)
        txt = f'E {x:,.0f}   N {y:,.0f}'
        if utm:
            txt += f'\n{lat[i]:.4f}°{"N" if lat[i] >= 0 else "S"}  {lon[i]:.4f}°E'
        ax.text(x + ox * pad, y + oy * pad, txt, fontsize=14, linespacing=1.5,
                ha='left' if ox > 0 else 'right',
                va='bottom' if oy > 0 else 'top')
    ax.text(0.5, 0.5, f'EPSG:{m["epsg"]}\n{m["name"]}\nunits: {m["units"]}',
            transform=ax.transAxes, ha='center', va='center',
            fontsize=20, linespacing=1.9)
    pad2 = (right - left) * 0.34
    ax.set_xlim(left - pad2, right + pad2); ax.set_ylim(bottom - pad2, top + pad2)
    ax.set_aspect('equal'); ax.axis('off')
    plt.show()


def show_pixel(path, col=None, row=None, steps=(240, 24)):
    """Zoom until one pixel fills the panel, and say how wide it is on the ground."""
    a, _ = _load(path)
    m = meta(path)
    W, H = m['width'], m['height']
    px, py = m['res']
    col = W // 2 if col is None else col
    row = H // 2 if row is None else row

    imgs = [_st(_thumb(path, (0, 1, 2), 600)).transpose(1, 2, 0)]
    wins = []
    for s in steps:
        c0 = int(np.clip(col - s // 2, 0, W - s)); r0 = int(np.clip(row - s // 2, 0, H - s))
        wins.append((c0, r0, s))
        imgs.append(_st(a[:3, r0:r0 + s, c0:c0 + s].astype(np.float32)).transpose(1, 2, 0))
    one = a[:, row, col]
    c0, r0, _ = wins[-1]
    rgb = imgs[-1][row - r0, col - c0]      # 화면과 같은 스트레치라야 색이 맞는다

    fig, axes = plt.subplots(1, len(imgs) + 1, figsize=(4.0 * (len(imgs) + 1), 4.8),
                             layout='constrained')
    scales = [(W, H)] + [(s, s) for _, _, s in wins]
    prev = (0, 0)
    for i, ax in enumerate(axes[:-1]):
        ax.imshow(imgs[i], interpolation='nearest')
        sw, sh = scales[i]
        k = imgs[i].shape[1] / sw
        if i < len(wins):
            cc, rr, s = wins[i]
            ax.add_patch(Rectangle(((cc - prev[0]) * k, (rr - prev[1]) * k),
                                   s * k, s * k, fill=False, ec=C_MARK, lw=2.6))
            prev = (cc, rr)
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
            color='white' if np.mean(rgb) < 0.5 else 'black')
    ax.set_xlim(-0.3, 1.3); ax.set_ylim(-0.3, 1.3)
    ax.set_aspect('equal'); ax.axis('off')
    plt.show()


# --------------------------------------------------------------- 2 and 3
def stats(path, bits=12):
    """Do min / max / mean / std fit the bit depth?"""
    a, _ = _load(path)
    ceiling = 2 ** bits - 1
    print(f'{"band":>5} {"min":>7} {"max":>7} {"mean":>8} {"std":>7} '
          f'{"px at ceiling":>14}')
    for i, v in enumerate(a.astype(np.float32), start=1):
        print(f'{i:5d} {v.min():7.0f} {v.max():7.0f} {v.mean():8.1f} {v.std():7.1f}'
              f' {int((v >= ceiling).sum()):14d}')
    print(f'\n{bits}-bit range is 0 - {ceiling}')


def preview(path, size=800, lo=2, hi=98, bits=12):
    """The flag map is built at full resolution, then reduced with `any`, so single
    odd pixels do not get averaged away."""
    full, _ = _load(path)
    a = full[:3].astype(np.float32)
    ceiling = 2 ** bits - 1
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
