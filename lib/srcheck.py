"""Checking an SR output against the input it came from, in three passes.

    1  structure    pixel count, band count, data type
    2  geometry     CRS, pixel size, origin, footprint
    3  radiometry   mean, standard deviation, histogram shape

The GeoTIFF reader is the one from the intake page, which uses rasterio.
"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from intake import meta, read

__all__ = ['structure', 'show_structure', 'geometry', 'show_geometry',
           'radiometry', 'show_radiometry', 'np', 'plt']

C_IN, C_OUT, C_BAD, C_OK = '#2b6cb0', '#f0a63a', '#c0392b', '#2e8b57'
NAMES = {4: ['B1 red', 'B2 green', 'B3 blue', 'B4 NIR'],
         3: ['B1 red', 'B2 green', 'B3 blue']}


def _n(c):
    return NAMES.get(c, [f'B{i}' for i in range(1, c + 1)])


def _mark(ok):
    return 'ok  ' if ok else 'CHECK'


# --------------------------------------------------------------- 1. structure
def structure(src, dst, scale=2):
    a, b = meta(src), meta(dst)
    rx, ry = b['width'] / a['width'], b['height'] / a['height']
    print(f'{"":12} {"input":>14} {"output":>14}   {"":>5}')
    print(f'{"width":12} {a["width"]:>14} {b["width"]:>14}   '
          f'{_mark(rx == scale)} x{rx:g}')
    print(f'{"height":12} {a["height"]:>14} {b["height"]:>14}   '
          f'{_mark(ry == scale)} x{ry:g}')
    print(f'{"bands":12} {a["count"]:>14} {b["count"]:>14}   '
          f'{_mark(a["count"] == b["count"])}')
    print(f'{"dtype":12} {a["dtype"]:>14} {b["dtype"]:>14}   '
          f'{_mark(a["dtype"] == b["dtype"])}')


def show_structure(src, dst, scale=2):
    """Pixel count, band count, data type — side by side."""
    a, b = meta(src), meta(dst)
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.6), layout='constrained')

    ax = axes[0]
    for m, col, z in ((b, C_OUT, 1), (a, C_IN, 2)):
        ax.add_patch(Rectangle((0, 0), m['width'], m['height'], fc=col, ec='#333',
                               lw=1.8, alpha=0.75, zorder=z))
    ax.text(a['width'] * 0.5, a['height'] * 0.5, f'{a["width"]} x {a["height"]}',
            ha='center', va='center', fontsize=18, zorder=3)
    ax.text(b['width'] * 0.98, b['height'] * 0.97, f'{b["width"]} x {b["height"]}',
            ha='right', va='top', fontsize=18, zorder=3)
    ax.set_xlim(-b['width'] * .05, b['width'] * 1.05)
    ax.set_ylim(b['height'] * 1.05, -b['height'] * .05)
    ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    ok = b['width'] == a['width'] * scale and b['height'] == a['height'] * scale
    ax.set_title('1.  pixels', fontsize=21, pad=12, loc='left')
    ax.set_xlabel(f'x{b["width"] / a["width"]:g}', fontsize=19,
                  color=C_OK if ok else C_BAD)

    ax = axes[1]
    for k, (m, col) in enumerate(((a, C_IN), (b, C_OUT))):
        for i, nm in enumerate(_n(m['count'])):
            ax.add_patch(Rectangle((k + 0.1, i), 0.8, 0.82, fc=col, ec='none',
                                   alpha=0.8))
            ax.text(k + 0.5, i + 0.41, nm, ha='center', va='center', fontsize=15)
    miss = a['count'] - b['count']
    for i in range(b['count'], a['count']):
        ax.add_patch(Rectangle((1.1, i), 0.8, 0.82, fc='none', ec=C_BAD, lw=2.4,
                               ls='--'))
    ax.set_xlim(0, 2); ax.set_ylim(-0.3, max(a['count'], b['count']) + 0.3)
    ax.set_xticks([0.5, 1.5]); ax.set_xticklabels(['input', 'output'], fontsize=17)
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title('2.  bands', fontsize=21, pad=12, loc='left')
    ax.set_xlabel(f'{a["count"]} -> {b["count"]}', fontsize=19,
                  color=C_OK if miss == 0 else C_BAD)

    ax = axes[2]
    top = float(np.iinfo(a['dtype']).max)
    for k, (m, col) in enumerate(((a, C_IN), (b, C_OUT))):
        ax.add_patch(Rectangle((0, k), top, 0.7, fc=col, ec='#333', lw=1.4,
                               alpha=0.8))
        ax.text(top * 0.5, k + 0.35, m['dtype'], ha='center', va='center',
                fontsize=19)
    ax.set_xlim(-top * .04, top * 1.04); ax.set_ylim(-0.3, 2.0)
    ax.set_yticks([0.35, 1.35]); ax.set_yticklabels(['input', 'output'], fontsize=17)
    ax.tick_params(labelsize=14)
    ax.set_xlabel('0 - %d' % top, fontsize=19,
                  color=C_OK if a['dtype'] == b['dtype'] else C_BAD)
    for sp in ('top', 'right', 'left'):
        ax.spines[sp].set_visible(False)
    ax.set_title('3.  data type', fontsize=21, pad=12, loc='left')
    plt.show()


# --------------------------------------------------------------- 2. geometry
def geometry(src, dst, scale=2):
    a, b = meta(src), meta(dst)
    same_crs = a['epsg'] == b['epsg']
    rx = a['res'][0] / b['res'][0]
    o_a, o_b = (a['bounds'][0], a['bounds'][3]), (b['bounds'][0], b['bounds'][3])
    d = max(abs(o_a[0] - o_b[0]), abs(o_a[1] - o_b[1]))
    print(f'{"CRS":12} {a["epsg"]:>16} {b["epsg"]:>16}   {_mark(same_crs)}')
    print(f'{"pixel size":12} {a["res"][0]:>16.6f} {b["res"][0]:>16.6f}   '
          f'{_mark(abs(rx - scale) < 1e-6)} /{rx:.4f}')
    print(f'{"origin E":12} {o_a[0]:>16.4f} {o_b[0]:>16.4f}')
    print(f'{"origin N":12} {o_a[1]:>16.4f} {o_b[1]:>16.4f}   '
          f'{_mark(d < 1e-6)} {d:.6f} m apart')
    for nm, i in (('left', 0), ('bottom', 1), ('right', 2), ('top', 3)):
        print(f'{"bound " + nm:12} {a["bounds"][i]:>16.3f} {b["bounds"][i]:>16.3f}')


def show_geometry(src, dst, grid=8):
    """Same ground, same corner, half the pixel."""
    a, b = meta(src), meta(dst)
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.8), layout='constrained')

    ax = axes[0]
    for m, col, lw in ((a, C_IN, 5.0), (b, C_OUT, 2.0)):
        l, bo, r, t = m['bounds']
        ax.add_patch(Rectangle((l, bo), r - l, t - bo, fc='none', ec=col, lw=lw))
    l, bo, r, t = a['bounds']
    ax.plot([l], [t], 'o', color=C_BAD, ms=11, zorder=3)
    ax.text(l, t, f'  E {l:,.1f}\n  N {t:,.1f}', fontsize=15, va='bottom',
            linespacing=1.5)
    m = (r - l) * 0.22
    ax.set_xlim(l - m, r + m); ax.set_ylim(bo - m, t + m)
    ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title('1.  footprint', fontsize=21, pad=12, loc='left')
    ax.set_xlabel(f'{r - l:,.1f} x {t - bo:,.1f} m', fontsize=19)

    for ax, m, col, k in ((axes[1], a, C_IN, 1), (axes[2], b, C_OUT, 2)):
        n = grid * k
        step = m['res'][0]
        for i in range(n + 1):
            ax.axhline(i * step, color=col, lw=1.4)
            ax.axvline(i * step, color=col, lw=1.4)
        ax.add_patch(Rectangle((0, 0), step, step, fc=col, ec='none', alpha=0.55))
        side = grid * a['res'][0]
        ax.set_xlim(-side * 0.05, side * 1.05); ax.set_ylim(-side * 0.05, side * 1.05)
        ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_title(f'{k + 1}.  {"input" if k == 1 else "output"} grid',
                     fontsize=21, pad=12, loc='left')
        ax.set_xlabel(f'{step:.3f} m per pixel', fontsize=19)
    plt.show()


# --------------------------------------------------------------- 3. radiometry
def radiometry(src, dst):
    a, b = read(src).astype(np.float32), read(dst).astype(np.float32)
    n = min(len(a), len(b))
    print(f'{"band":>5} {"mean in":>9} {"mean out":>9} {"d":>7}   '
          f'{"std in":>8} {"std out":>8} {"d%":>7}   {"min":>13} {"max":>15}')
    for i in range(n):
        u, v = a[i], b[i]
        print(f'{i + 1:5d} {u.mean():9.2f} {v.mean():9.2f} {v.mean() - u.mean():7.2f}   '
              f'{u.std():8.2f} {v.std():8.2f} {100 * (v.std() / u.std() - 1):6.1f}%   '
              f'{u.min():6.0f}->{v.min():6.0f} {u.max():7.0f}->{v.max():7.0f}')


def show_radiometry(src, dst, bins=240):
    """Histogram shape, band by band, then what moved."""
    a, b = read(src).astype(np.float32), read(dst).astype(np.float32)
    n = min(len(a), len(b))
    top = max(a[:n].max(), b[:n].max())
    fig, axes = plt.subplots(1, n, figsize=(5.2 * n, 5.4), layout='constrained')
    edges = np.linspace(0, top, bins)
    for i in range(n):
        ax = np.atleast_1d(axes)[i]
        for v, col, nm in ((a[i], C_IN, 'input'), (b[i], C_OUT, 'output')):
            ax.hist(v.ravel(), bins=edges, histtype='step', lw=2.4, color=col,
                    label=nm, weights=np.full(v.size, 1.0 / v.size))
        ax.set_yscale('log'); ax.set_yticks([]); ax.set_xlim(0, top)
        ax.tick_params(labelsize=14)
        ax.set_title(f'{i + 1}.  {_n(n)[i]}', fontsize=21, pad=12, loc='left')
        ax.set_xlabel(f'mean {b[i].mean() - a[i].mean():+.2f}      '
                      f'std {100 * (b[i].std() / a[i].std() - 1):+.1f}%\n'
                      f'max {a[i].max():.0f} -> {b[i].max():.0f}',
                      fontsize=17, linespacing=1.6)
        ax.legend(fontsize=15, frameon=False)
        for sp in ('top', 'right', 'left'):
            ax.spines[sp].set_visible(False)
    plt.show()
