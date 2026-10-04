"""What the pixel values look like, and whether the scene is worth using.

    distribution / show_distribution   where the values sit
    condition    / show_condition      cloud and sharpness, two scenes compared

Reading is done by the intake helper, which uses rasterio.
"""
import matplotlib.pyplot as plt
import numpy as np

from intake import _st, meta, read

__all__ = ['distribution', 'show_distribution', 'condition', 'show_condition',
           'np', 'plt']

C_IN, C_BAD, C_OK = '#2b6cb0', '#c0392b', '#2e8b57'
PCTS = (2, 50, 98)
NAMES = {4: ['B1 red', 'B2 green', 'B3 blue', 'B4 NIR'],
         3: ['B1 red', 'B2 green', 'B3 blue']}


def _names(n):
    return NAMES.get(n, [f'B{i}' for i in range(1, n + 1)])


def _sharp(g):
    """Mean gradient divided by mean level — brightness cancels out, so a dark
    scene and a bright one can be compared."""
    d = np.abs(np.diff(g, axis=0))[:, :-1] + np.abs(np.diff(g, axis=1))[:-1]
    return float(d.mean() / g.mean())


# --------------------------------------------------------------- distribution
def distribution(path):
    a = read(path).astype(np.float32)
    print(f'{"band":>8} {"min":>7} {"p2":>7} {"p50":>7} {"p98":>7} {"max":>7}'
          f' {"mean":>8} {"std":>7}')
    for i, v in enumerate(a):
        p = np.percentile(v, PCTS)
        print(f'{_names(len(a))[i]:>8} {v.min():7.0f} {p[0]:7.0f} {p[1]:7.0f}'
              f' {p[2]:7.0f} {v.max():7.0f} {v.mean():8.1f} {v.std():7.1f}')


def show_distribution(path, bits=12, bins=260):
    """Where the values sit, and how much of the range they leave empty."""
    a = read(path).astype(np.float32)
    ceil = 2 ** bits - 1
    names = _names(len(a))
    edges = np.linspace(0, ceil, bins)

    fig, axes = plt.subplots(1, 2, figsize=(15.6, 5.6), layout='constrained')
    ax = axes[0]
    for i, v in enumerate(a):
        ax.hist(v.ravel(), bins=edges, histtype='step', lw=2.4,
                weights=np.full(v.size, 1.0 / v.size), label=names[i])
    p = np.percentile(a, PCTS)
    for x in p:                      # 값은 오른쪽 누적 곡선에만 적는다 — 여기선 겹친다
        ax.axvline(x, color='#666', ls='--', lw=1.6)
    ax.set_yscale('log'); ax.set_yticks([]); ax.set_xlim(0, ceil)
    ax.tick_params(labelsize=15)
    ax.set_xlabel('pixel value', fontsize=18)
    ax.legend(fontsize=15, frameon=False)
    ax.set_title('1.  where the values sit', fontsize=21, pad=12, loc='left')

    ax = axes[1]
    v = np.sort(a.ravel())
    y = np.linspace(0, 100, len(v))
    ax.plot(v, y, lw=3.0, color=C_IN)
    for q, x in zip(PCTS, p):
        ax.plot([x, x, 0], [0, q, q], color='#666', ls='--', lw=1.6)
        ax.text(x, q, f'   p{q} = {x:.0f}', fontsize=16, va='center', color='#444')
    ax.set_xlim(0, ceil); ax.set_ylim(0, 102)
    ax.tick_params(labelsize=15)
    ax.set_xlabel('pixel value', fontsize=18)
    ax.set_ylabel('% of pixels below', fontsize=18)
    ax.set_title('2.  how fast it gets there', fontsize=21, pad=12, loc='left')
    for ax in axes:
        for sp in ('top', 'right'):
            ax.spines[sp].set_visible(False)
    plt.show()


# --------------------------------------------------------------- condition
def condition(paths, labels, bright=900):
    """Indicators, not a cloud mask — but they separate a usable scene from one
    that is covered."""
    print(f'{"":>10} {"mean":>8} {"std":>8} {"p50":>7} {"p99":>7}'
          f' {"> " + str(bright):>8} {"sharpness":>10}')
    for p, lab in zip(paths, labels):
        g = read(p)[:3].astype(np.float32).mean(0)
        print(f'{lab:>10} {g.mean():8.1f} {g.std():8.1f} {np.percentile(g, 50):7.0f}'
              f' {np.percentile(g, 99):7.0f} {100 * (g > bright).mean():7.1f}%'
              f' {_sharp(g):10.4f}')


def show_condition(paths, labels, bright=900):
    """Top row is the scene, bottom row marks everything brighter than a cut."""
    n = len(paths)
    fig, axes = plt.subplots(2, n, figsize=(5.4 * n, 10.4), layout='constrained')
    axes = np.atleast_2d(axes)
    for k, (p, lab) in enumerate(zip(paths, labels)):
        a = read(p)[:3].astype(np.float32)
        g = a.mean(0)
        img = _st(a).transpose(1, 2, 0)
        hot = g > bright
        flag = img.copy()
        flag[hot] = (1, 0, 0)
        axes[0, k].imshow(img)
        axes[0, k].set_title(f'{k + 1}.  {lab}', fontsize=22, pad=12, loc='left')
        axes[0, k].set_xlabel(f'sharpness {_sharp(g):.4f}', fontsize=19)
        axes[1, k].imshow(flag)
        axes[1, k].set_title(f'{n + k + 1}.  {lab}, over {bright}', fontsize=22,
                             pad=12, loc='left')
        axes[1, k].set_xlabel(f'{100 * hot.mean():.1f}% of the scene', fontsize=19,
                              color=C_BAD if hot.mean() > 0.05 else C_OK)
    for ax in axes.ravel():
        ax.set_xticks([]); ax.set_yticks([]); ax.set_box_aspect(1)
    plt.show()
