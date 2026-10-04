"""Reading a satellite image properly — four questions, one answer each.

    check_read    am I reading it correctly?
    check_type    what type are the numbers?
    check_size    how big is it?
    check_bands   how are the bands arranged?
    check_order   what a guessed band order looks like

The GeoTIFF reader is the one from the intake page, which uses rasterio.
"""
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from intake import _st, meta, read

__all__ = ['check_read', 'check_type', 'check_size', 'check_bands',
           'check_order', 'np', 'plt']

C_IN, C_BAD, C_OK = '#2b6cb0', '#c0392b', '#2e8b57'
NAMES = {4: ['B1  red', 'B2  green', 'B3  blue', 'B4  near infrared'],
         3: ['B1  red', 'B2  green', 'B3  blue']}


def _small(path, size=520):
    a = read(path)
    k = max(1, min(a.shape[1], a.shape[2]) // size)
    return a[:, ::k, ::k].astype(np.float32)


def _names(n):
    return NAMES.get(n, [f'B{i}' for i in range(1, n + 1)])


# --------------------------------------------------------------- read
def check_read(path):
    """Take bands 1, 2, 3 and put them on screen.  Does an image come out?"""
    a = _small(path)
    m = meta(path)
    top = np.iinfo(m['dtype']).max
    asis = np.clip(a[:3] / top, 0, 1).transpose(1, 2, 0)
    good = _st(a[:3]).transpose(1, 2, 0)

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 6.0), layout='constrained')
    for ax, img, t, lab, col in (
            (axes[0], asis, '1.  as read', f'drawn on 0 - {top:,}', C_BAD),
            (axes[1], good, '2.  rescaled', f'drawn on {a[:3].min():.0f} - '
                                            f'{a[:3].max():.0f}', C_OK)):
        ax.imshow(img)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_box_aspect(1)
        ax.set_title(t, fontsize=22, pad=12, loc='left')
        ax.set_xlabel(lab, fontsize=19, color=col)
        for sp in ax.spines.values():
            sp.set(color=col, linewidth=3.0)
    plt.show()


def check_order(path):
    """The same three bands, taken in both orders."""
    a = _small(path)
    good = _st(a[:3]).transpose(1, 2, 0)

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 6.0), layout='constrained')
    for ax, img, t, lab, col in (
            (axes[0], good, '1.  B1 B2 B3', 'red, green, blue', C_OK),
            (axes[1], good[:, :, ::-1], '2.  B3 B2 B1', 'reversed', C_BAD)):
        ax.imshow(img)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_box_aspect(1)
        ax.set_title(t, fontsize=22, pad=12, loc='left')
        ax.set_xlabel(lab, fontsize=19, color=col)
        for sp in ax.spines.values():
            sp.set(color=col, linewidth=3.0)
    plt.show()


# --------------------------------------------------------------- bands
def check_bands(path):
    """Each band, then the two composites that prove which one is which."""
    a = _small(path)
    n = len(a)
    names = _names(n)
    cols = n + (2 if n >= 4 else 1)
    fig, axes = plt.subplots(1, cols, figsize=(3.3 * cols, 4.4), layout='constrained')
    for i in range(n):
        axes[i].imshow(_st(a[i]), cmap='gray')
        axes[i].set_title(names[i], fontsize=17, pad=8)
    axes[n].imshow(_st(a[:3]).transpose(1, 2, 0))
    axes[n].set_title('B1 B2 B3', fontsize=17, pad=8)
    axes[n].set_xlabel('true colour', fontsize=16)
    if n >= 4:
        axes[n + 1].imshow(_st(a[[3, 0, 1]]).transpose(1, 2, 0))
        axes[n + 1].set_title('B4 B1 B2', fontsize=17, pad=8)
        axes[n + 1].set_xlabel('NIR as red', fontsize=16)
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([]); ax.set_box_aspect(1)
    plt.show()


# --------------------------------------------------------------- size
def check_size(path):
    """Pixels, ground, and the two very different numbers for 'size'."""
    m = meta(path)
    w, h, c = m['width'], m['height'], m['count']
    px, py = m['res']
    disk = os.path.getsize(path) / 2 ** 20
    ram = w * h * c * np.dtype(m['dtype']).itemsize / 2 ** 20

    fig, axes = plt.subplots(1, 2, figsize=(14.6, 5.8), layout='constrained',
                             gridspec_kw=dict(width_ratios=[1, 1.1]))
    ax = axes[0]
    ax.add_patch(Rectangle((0, 0), w, h, fc='#dce6f2', ec=C_IN, lw=2.5))
    ax.annotate('', xy=(0, h * 1.05), xytext=(w, h * 1.05),
                arrowprops=dict(arrowstyle='<->', color=C_IN, lw=1.8))
    ax.text(w / 2, h * 1.07, f'{w:,} px   =   {w * px:,.0f} m', ha='center',
            va='bottom', fontsize=18, color=C_IN)
    ax.text(w / 2, h / 2, f'{w * h / 1e6:.1f} M pixels\n{c} bands',
            ha='center', va='center', fontsize=21, linespacing=1.8)
    ax.text(-w * 0.04, h / 2, f'{h:,} px   =   {h * py:,.0f} m', ha='right',
            va='center', rotation=90, fontsize=18, color=C_IN)
    ax.set_xlim(-w * 0.2, w * 1.06); ax.set_ylim(h * 1.2, -h * 0.06)
    ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title('1.  on the ground', fontsize=21, pad=12, loc='left')

    ax = axes[1]
    ax.barh(['on disk', 'in memory'], [disk, ram], color=[C_IN, C_BAD], height=0.55)
    for i, v in enumerate((disk, ram)):
        ax.text(v, i, f'  {v:,.0f} MB', va='center', fontsize=20)
    ax.set_xlim(0, ram * 1.3)
    ax.tick_params(labelsize=18)
    ax.set_xlabel('MB', fontsize=18)
    for sp in ('top', 'right', 'left'):
        ax.spines[sp].set_visible(False)
    ax.set_title('2.  as a file', fontsize=21, pad=12, loc='left')
    plt.show()


# --------------------------------------------------------------- type
def check_type(path, bits=12):
    """The type says how big the box is, not how full it is."""
    a = read(path).astype(np.float32)
    m = meta(path)
    top = float(np.iinfo(m['dtype']).max)
    ceil = 2 ** bits - 1
    lo, hi = float(a.min()), float(a.max())

    fig, axes = plt.subplots(1, 2, figsize=(15.5, 5.2), layout='constrained',
                             gridspec_kw=dict(width_ratios=[1, 1.2]))
    ax = axes[0]
    ax.add_patch(Rectangle((0, 0.3), top, 0.4, fc='#e9edf2', ec='#99a3af', lw=1.4))
    ax.add_patch(Rectangle((lo, 0.3), hi - lo, 0.4, fc=C_IN, ec='none'))
    ax.text(top, 0.92, f'{m["dtype"]}   0 - {top:,.0f}', fontsize=17, ha='right')
    ax.text(0, 0.16, f'actually   {lo:,.0f} - {hi:,.0f}', fontsize=17, color=C_IN)
    ax.set_xlim(-top * 0.03, top * 1.05); ax.set_ylim(0.05, 1.05)
    ax.set_yticks([]); ax.tick_params(labelsize=14)
    ax.set_xlabel('pixel value', fontsize=18)
    ax.set_title('1.  the box', fontsize=21, pad=12, loc='left')
    for sp in ('top', 'right', 'left'):
        ax.spines[sp].set_visible(False)

    ax = axes[1]
    for i, v in enumerate(a):
        ax.hist(v.ravel(), bins=np.linspace(0, ceil, 260), histtype='step', lw=2.2,
                weights=np.full(v.size, 1.0 / v.size), label=_names(len(a))[i])
    ax.set_yscale('log'); ax.set_yticks([]); ax.set_xlim(0, ceil)
    ax.tick_params(labelsize=14)
    ax.set_xlabel('pixel value', fontsize=18)
    ax.legend(fontsize=15, frameon=False)
    ax.set_title(f'2.  what is inside   ({bits}-bit range)', fontsize=21, pad=12,
                 loc='left')
    for sp in ('top', 'right', 'left'):
        ax.spines[sp].set_visible(False)
    plt.show()
