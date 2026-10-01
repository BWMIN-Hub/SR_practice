"""12-bit tiles -> 8-bit: how the window is chosen, and what it costs.

One helper so the notebook cells stay short.
"""
import matplotlib.pyplot as plt
import numpy as np

RULE = (150.0, 800.0)
C_WIN = ['#8e44ad', '#2b6cb0', '#2e8b57']
C_TILE, C_CLOUD, C_CUT = '#9fb6cc', '#e8c33a', (1.0, 0.12, 0.12)

__all__ = ['RULE', 'load', 'windows', 'show', 'np', 'plt']


def load(path):
    """-> dict with three uint16 tiles and one array of cloud pixel values."""
    z = np.load(path)
    return {k: z[k].astype(np.float32) for k in z.files}


def windows(pool, specs):
    """specs: ('pct', lo, hi) or ('rule',)  ->  [(label, (lo, hi) per band), ...]

    pool is (N, 3) or (H, W, 3): the window is found for each band separately,
    which is what real tools do and what keeps the colour intact.
    """
    pool = np.asarray(pool, np.float32).reshape(-1, 3)
    out = []
    for s in specs:
        if s[0] == 'rule':
            lo = np.full(3, RULE[0]); hi = np.full(3, RULE[1])
            out.append((f'rule {RULE[0]:.0f} - {RULE[1]:.0f}', (lo, hi)))
        else:
            lo = np.percentile(pool, s[1], axis=0)
            hi = np.percentile(pool, s[2], axis=0)
            out.append((f'{s[1]:g} - {s[2]:g}%', (lo, hi)))
    return out


def show(tile, wins, cloud=None, xmax=1000, size=4.6, band=0):
    """Top: the tile under each window, cut pixels in red.  Bottom: the histogram.

    Each band gets its own window.  The histogram below shows one band, with that
    band's bounds drawn on it.
    """
    b = tile[..., band]
    fig = plt.figure(figsize=(size * len(wins), size * 1.75))
    gs = fig.add_gridspec(2, len(wins), height_ratios=[1.55, 1.0], hspace=0.22,
                          wspace=0.06)

    for i, ((lab, (lo, hi)), col) in enumerate(zip(wins, C_WIN)):
        ax = fig.add_subplot(gs[0, i])
        img = np.clip((tile - lo) / (hi - lo), 0, 1)
        sat = ((tile <= lo) | (tile >= hi)).any(axis=2)
        img[sat] = C_CUT
        ax.imshow(img)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_box_aspect(1)
        ax.set_title(f'{lab}\n{lo[band]:.0f} - {hi[band]:.0f}', fontsize=15, color=col,
                     linespacing=1.3)
        cut = 100 * sat.mean()
        q = np.percentile(tile.reshape(-1, 3), (2, 98), axis=0)
        used = np.clip((q - lo) / (hi - lo), 0, 1)
        ax.set_xlabel(f'{cut:.1f}% cut     uses {(used[1] - used[0]).mean() * 255:.0f}'
                      f' of 255', fontsize=13)
        for sp in ax.spines.values():
            sp.set(color=col, linewidth=2.2)

    ax = fig.add_subplot(gs[1, :])
    bins = np.linspace(0, xmax, 321)
    if cloud is not None:
        ax.hist(cloud, bins=bins, color=C_CLOUD, zorder=1)
    ax.hist(b.ravel(), bins=bins, color=C_TILE, zorder=2)
    top = ax.get_ylim()[1]
    for k, ((lab, (lo_a, hi_a)), col) in enumerate(zip(wins, C_WIN)):
        lo, hi = lo_a[band], hi_a[band]
        y = top * (0.93 - 0.11 * k)
        for v in (lo, hi):
            ax.axvline(v, color=col, lw=2.0)
        ax.annotate('', xy=(lo, y), xytext=(hi, y),
                    arrowprops=dict(arrowstyle='<->', color=col, lw=1.8))
        ax.text((lo + hi) / 2, y + top * 0.02, lab, color=col, fontsize=14,
                ha='center', va='bottom')
    ax.set_xlim(0, xmax); ax.set_yticks([])
    ax.set_xlabel(f'pixel value  (band {band + 1})', fontsize=14)
    for sp in ('top', 'right', 'left'):
        ax.spines[sp].set_visible(False)
    plt.show()
