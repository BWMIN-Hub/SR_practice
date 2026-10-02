"""Tiled SR inference — the knobs that decide whether it runs at all, and how fast.

A whole scene does not fit on a GPU, so it is cut into tiles, each tile is widened
by a margin, the margins are thrown away after the forward pass, and the pieces are
put back together.  Four numbers decide everything: tile, overlap, batch, precision.

    from infer import Config, sr_tiled, show_tiles, show_overlap, show_batch
"""
import time
from dataclasses import dataclass, field, replace

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Rectangle

__all__ = ['Config', 'plan', 'sr_tiled', 'sr_tiled_buggy', 'show_tiles', 'show_overlap',
           'show_batch', 'show_precision', 'bench', 'show_bench', 'seam_scan',
           'show_seam', 'show_fail', 'try_run', 'show_result', 'memory_law',
           'show_memory_law', 'replace',
           'np', 'plt', 'torch']

C_READ, C_KEEP, C_EDGE = '#f0a63a', '#2b6cb0', '#c0392b'
# 7색 — 타일 열 수(8, 16 …)와 어긋나게 두어야 줄무늬로 보이지 않는다
PAL = ['#2b6cb0', '#f0a63a', '#2e8b57', '#c0392b', '#8e44ad', '#17a2b8', '#d4a017']


@dataclass
class Config:
    """Everything the run needs, in one place."""
    scale: int = 2
    tile: int = 256          # tile side, in input pixels
    overlap: int = 16        # margin added on every side, thrown away afterwards
    batch: int = 4           # tiles sent to the GPU in one forward pass
    precision: str = 'fp32'  # 'fp32' or 'fp16'
    device: str = field(
        default_factory=lambda: 'cuda' if torch.cuda.is_available() else 'cpu')

    @property
    def read(self):
        """Side of the patch the network actually sees."""
        return self.tile + 2 * self.overlap


# --------------------------------------------------------------- geometry
def plan(h, w, cfg):
    """Where the tiles sit, and what it costs.  No pixels are touched here."""
    ny, nx = -(-h // cfg.tile), -(-w // cfg.tile)
    n = ny * nx
    seen = n * cfg.read ** 2
    return dict(ny=ny, nx=nx, tiles=n, read=cfg.read,
                calls=-(-n // cfg.batch),
                # 타일이 batch 보다 적으면 그만큼만 간다 — 설정값이 아니라 실제 양
                px_per_call=min(cfg.batch, n) * cfg.read ** 2,
                redundancy=seen / (h * w) - 1,
                padded=(ny * cfg.tile + 2 * cfg.overlap,
                        nx * cfg.tile + 2 * cfg.overlap))


def _pad(a, cfg, h, w):
    """Reflect-pad so every tile is exactly `read` wide — including the last row
    and column, which would otherwise come out short."""
    p = plan(h, w, cfg)
    bottom = cfg.overlap + p['ny'] * cfg.tile - h
    right = cfg.overlap + p['nx'] * cfg.tile - w
    return np.pad(a, ((cfg.overlap, bottom), (cfg.overlap, right), (0, 0)), mode='reflect')


# --------------------------------------------------------------- running
def sr_tiled(img, net, cfg, report=False):
    """Tile, run, crop, stitch.  Returns the SR image at scale x the input."""
    h, w = img.shape[:2]
    p = plan(h, w, cfg)
    src = _pad(img, cfg, h, w)
    out = np.empty((p['ny'] * cfg.tile * cfg.scale,
                    p['nx'] * cfg.tile * cfg.scale, 3), np.uint8)
    keep = cfg.tile * cfg.scale
    edge = cfg.overlap * cfg.scale          # crop in OUTPUT pixels, not input ones
    pos = [(iy, ix) for iy in range(p['ny']) for ix in range(p['nx'])]

    if cfg.device.startswith('cuda'):
        torch.cuda.reset_peak_memory_stats(cfg.device)
        torch.cuda.synchronize(cfg.device)
    t0 = time.time()
    for i in range(0, len(pos), cfg.batch):
        chunk = pos[i:i + cfg.batch]
        x = np.stack([src[iy * cfg.tile:iy * cfg.tile + cfg.read,
                          ix * cfg.tile:ix * cfg.tile + cfg.read] for iy, ix in chunk])
        x = torch.from_numpy(x).permute(0, 3, 1, 2).float().to(cfg.device)
        with torch.no_grad():
            if cfg.precision == 'fp16':
                with torch.autocast(cfg.device.split(':')[0], dtype=torch.float16):
                    y = net(x)
            else:
                y = net(x)
        y = y.float().clamp(0, 255).round().byte().permute(0, 2, 3, 1).cpu().numpy()
        for (iy, ix), tile in zip(chunk, y):
            out[iy * keep:(iy + 1) * keep, ix * keep:(ix + 1) * keep] = \
                tile[edge:edge + keep, edge:edge + keep]
        del x, y
    if cfg.device.startswith('cuda'):
        torch.cuda.synchronize(cfg.device)
    dt = time.time() - t0
    vram = (torch.cuda.max_memory_allocated(cfg.device) / 2 ** 30
            if cfg.device.startswith('cuda') else 0.0)
    out = out[:h * cfg.scale, :w * cfg.scale]       # back to the size that was asked for
    if report:
        print(f'{p["tiles"]} tiles in {p["calls"]} calls   {dt:.2f} s   '
              f'{vram:.2f} GiB   out {out.shape[1]} x {out.shape[0]}')
    return out


def sr_tiled_buggy(img, net, cfg, bug):
    """The same routine with one line wrong.  Returns whatever shape it produces.

    'no_pad'     the image is not padded up to a whole number of tiles
    'input_crop' the margin is cropped in input pixels instead of output pixels
    'no_trim'    the padding is never trimmed off the result
    """
    h, w = img.shape[:2]
    p = plan(h, w, cfg)
    src = img if bug == 'no_pad' else _pad(img, cfg, h, w)
    keep = cfg.tile * cfg.scale
    edge = cfg.overlap if bug == 'input_crop' else cfg.overlap * cfg.scale
    out = np.zeros((p['ny'] * keep, p['nx'] * keep, 3), np.uint8)
    for iy in range(p['ny']):
        for ix in range(p['nx']):
            patch = src[iy * cfg.tile:iy * cfg.tile + cfg.read,
                        ix * cfg.tile:ix * cfg.tile + cfg.read]
            if patch.shape[0] < 8 or patch.shape[1] < 8:
                continue
            x = torch.from_numpy(patch[None]).permute(0, 3, 1, 2).float().to(cfg.device)
            with torch.no_grad():
                y = net(x)[0].clamp(0, 255).round().byte().permute(1, 2, 0).cpu().numpy()
            cut = y[edge:edge + keep, edge:edge + keep]
            out[iy * keep:iy * keep + cut.shape[0],
                ix * keep:ix * keep + cut.shape[1]] = cut
    return out if bug == 'no_trim' else out[:h * cfg.scale, :w * cfg.scale]


# --------------------------------------------------------------- pictures
def _grid(ax, h, w, cfg, show_margin=True, colour=None, groups=None):
    p = plan(h, w, cfg)
    for k, (iy, ix) in enumerate([(a, b) for a in range(p['ny']) for b in range(p['nx'])]):
        x0, y0 = ix * cfg.tile, iy * cfg.tile
        if show_margin and cfg.overlap:
            ax.add_patch(Rectangle((x0 - cfg.overlap, y0 - cfg.overlap),
                                   cfg.read, cfg.read, fc=C_READ, ec='none',
                                   alpha=0.28, zorder=1))
        fc = 'none'
        if groups is not None:
            fc = PAL[(k // groups) % len(PAL)]
        ax.add_patch(Rectangle((x0, y0), cfg.tile, cfg.tile, fc=fc,
                               ec=colour or C_KEEP, lw=1.6, alpha=1.0 if fc == 'none' else 0.75,
                               zorder=3))
    ax.add_patch(Rectangle((0, 0), w, h, fc='none', ec='#333', lw=2.0, zorder=4))
    m = max(h, w) * 0.07
    ax.set_xlim(-m, w + m); ax.set_ylim(h + m, -m)
    ax.set_aspect('equal')
    ax.set_xticks([]); ax.set_yticks([])          # axis('off') 를 쓰면 xlabel 도 사라진다
    for sp in ax.spines.values():
        sp.set_visible(False)
    return p


def show_tiles(shape, tiles=(128, 256, 512), cfg=None):
    """Same image, same margin, three tile sizes."""
    h, w = shape[:2]
    cfg = cfg or Config()
    fig, axes = plt.subplots(1, len(tiles), figsize=(4.6 * len(tiles), 5.2),
                             layout='constrained')
    for i, (ax, t) in enumerate(zip(np.atleast_1d(axes), tiles), start=1):
        p = _grid(ax, h, w, replace(cfg, tile=t))
        ax.set_title(f'{i}.  tile {t}', fontsize=21, pad=12, loc='left')
        ax.set_xlabel(f'{p["tiles"]} tiles     +{100 * p["redundancy"]:.0f}% pixels',
                      fontsize=18)
    plt.show()


def show_overlap(shape, overlaps=(0, 16, 64), cfg=None):
    """Same tile size, three margins.  Orange is read and thrown away."""
    h, w = shape[:2]
    cfg = cfg or Config()
    fig, axes = plt.subplots(1, len(overlaps), figsize=(4.6 * len(overlaps), 5.2),
                             layout='constrained')
    for i, (ax, o) in enumerate(zip(np.atleast_1d(axes), overlaps), start=1):
        c = replace(cfg, overlap=o)
        p = _grid(ax, h, w, c)
        ax.set_title(f'{i}.  overlap {o}', fontsize=21, pad=12, loc='left')
        ax.set_xlabel(f'reads {c.read} px     +{100 * p["redundancy"]:.0f}% pixels',
                      fontsize=18)
    plt.show()


def show_batch(shape, batches=(1, 4, 16), cfg=None):
    """Which tiles travel to the GPU together.  One colour is one forward pass."""
    h, w = shape[:2]
    cfg = cfg or Config()
    fig, axes = plt.subplots(1, len(batches), figsize=(4.6 * len(batches), 5.2),
                             layout='constrained')
    for i, (ax, b) in enumerate(zip(np.atleast_1d(axes), batches), start=1):
        c = replace(cfg, batch=b)
        p = _grid(ax, h, w, c, show_margin=False, groups=b)
        ax.set_title(f'{i}.  batch {b}', fontsize=21, pad=12, loc='left')
        ax.set_xlabel(f'{p["calls"]} calls     '
                      f'{p["px_per_call"] / 1e6:.2f} M px per call', fontsize=18)
    plt.show()


def show_precision(a, b, cfg=None, zoom=160):
    """fp32, fp16, and the difference between them."""
    d = np.abs(a.astype(np.int16) - b.astype(np.int16)).max(axis=2)
    y, x = np.unravel_index(d.argmax(), d.shape)
    y = int(np.clip(y - zoom // 2, 0, d.shape[0] - zoom))
    x = int(np.clip(x - zoom // 2, 0, d.shape[1] - zoom))
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.8), layout='constrained')
    axes[0].imshow(a[y:y + zoom, x:x + zoom])
    axes[1].imshow(b[y:y + zoom, x:x + zoom])
    im = axes[2].imshow(d[y:y + zoom, x:x + zoom], cmap='inferno', vmin=0,
                        vmax=max(d.max(), 1))
    for ax, t in zip(axes, ('1.  fp32', '2.  fp16',
                            f'3.  difference    max {d.max()}    '
                            f'mean {d.mean():.3f}')):
        ax.set_xticks([]); ax.set_yticks([]); ax.set_box_aspect(1)
        ax.set_title(t, fontsize=20, pad=10, loc='left')
    fig.colorbar(im, ax=axes[2], fraction=0.046)
    plt.show()


def bench(img, net, cfgs, warmup=True):
    """Run each config once and record seconds and peak VRAM.  OOM is recorded,
    not raised, so one bad row does not stop the table."""
    rows = []
    for c in cfgs:
        if warmup:
            try:
                sr_tiled(img[:c.tile, :c.tile], net, c)
            except Exception:
                pass
        try:
            if c.device.startswith('cuda'):
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(c.device)
                torch.cuda.synchronize(c.device)
            t0 = time.time()
            sr_tiled(img, net, c)
            if c.device.startswith('cuda'):
                torch.cuda.synchronize(c.device)
            rows.append(dict(cfg=c, seconds=time.time() - t0,
                             vram=(torch.cuda.max_memory_allocated(c.device) / 2 ** 30
                                   if c.device.startswith('cuda') else 0.0),
                             ok=True))
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            rows.append(dict(cfg=c, seconds=np.nan, vram=np.nan, ok=False))
    return rows


def show_bench(rows, by='tile'):
    """Seconds and peak VRAM, side by side.  A failed run is drawn as OOM."""
    lab = [str(getattr(r['cfg'], by)) for r in rows]
    sec = [r['seconds'] for r in rows]
    mem = [r['vram'] for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.4), layout='constrained')
    for ax, v, t, unit in ((axes[0], sec, '1.  runtime', 's'),
                           (axes[1], mem, '2.  peak VRAM', 'GiB')):
        good = [0 if np.isnan(x) else x for x in v]
        bars = ax.bar(lab, good, color=[C_KEEP if r['ok'] else C_EDGE for r in rows])
        for bar, x, r in zip(bars, v, rows):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    'OOM' if not r['ok'] else f'{x:.2f}', ha='center', va='bottom',
                    fontsize=16, color=C_EDGE if not r['ok'] else '#222')
        ax.set_title(t, fontsize=20, pad=10, loc='left')
        ax.set_xlabel(by, fontsize=18); ax.set_ylabel(unit, fontsize=18)
        ax.tick_params(labelsize=15)
        ax.margins(y=0.18)
        for sp in ('top', 'right'):
            ax.spines[sp].set_visible(False)
    plt.show()


# --------------------------------------------------------------- correctness
def seam_scan(img, net, cfg, overlaps=(0, 4, 8, 16, 32), shave=12):
    """Tiled output against one whole-image forward pass.  Only works on an image
    small enough to run in one go — that is the point of the comparison."""
    x = torch.from_numpy(img[None]).permute(0, 3, 1, 2).float().to(cfg.device)
    with torch.no_grad():
        ref = net(x)[0].clamp(0, 255).round().byte().permute(1, 2, 0).cpu().numpy()
    del x
    rows = []
    for o in overlaps:
        out = sr_tiled(img, net, replace(cfg, overlap=o))
        d = np.abs(out.astype(np.int16) - ref.astype(np.int16)).max(2)
        inner = d[shave:-shave, shave:-shave]
        rows.append(dict(overlap=o, peak=int(inner.max()),
                         mean=float(inner.mean()), err=d))
    return ref, rows


def show_seam(rows, show=(0, 8, 32)):
    """Where the tiled result differs from the whole-image result, and how much."""
    pick = [r for r in rows if r['overlap'] in show]
    n = len(pick)
    # 드문 최댓값에 맞추면 전부 까맣게 나온다 — 분포 윗꼬리에 맞춘다
    vmax = max(np.percentile(r['err'], 99.9) for r in pick) or 1
    fig, axes = plt.subplots(1, n + 1, figsize=(4.6 * (n + 1), 5.0),
                             layout='constrained')
    for i, (ax, r) in enumerate(zip(axes[:n], pick), start=1):
        im = ax.imshow(r['err'], cmap='inferno', vmin=0, vmax=vmax)
        ax.set_title(f'{i}.  overlap {r["overlap"]}', fontsize=20, pad=10, loc='left')
        ax.set_xlabel(f'peak {r["peak"]}    mean {r["mean"]:.3f}', fontsize=17)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_box_aspect(1)
    fig.colorbar(im, ax=axes[n - 1], fraction=0.046)
    ax = axes[n]
    ax.plot([r['overlap'] for r in rows], [r['mean'] for r in rows], 'o-',
            lw=2.6, ms=9, color=C_KEEP)
    ax.set_title(f'{n + 1}.  mean error', fontsize=20, pad=10, loc='left')
    ax.set_xlabel('overlap', fontsize=17); ax.set_ylabel('levels', fontsize=17)
    ax.tick_params(labelsize=15); ax.grid(alpha=0.25); ax.set_box_aspect(1)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    plt.show()


def show_fail(img, net, cfg, bugs=('no_pad', 'input_crop', 'no_trim')):
    """One line wrong in the crop, and the result is the wrong size or wrong."""
    h, w = img.shape[:2]
    want = (h * cfg.scale, w * cfg.scale)
    good = sr_tiled(img, net, cfg)
    panels = [('correct', good)]
    for b in bugs:
        panels.append((b, sr_tiled_buggy(img, net, cfg, b)))
    fig, axes = plt.subplots(1, len(panels), figsize=(4.4 * len(panels), 5.2),
                             layout='constrained')
    for i, (ax, (name, out)) in enumerate(zip(axes, panels), start=1):
        size_ok = out.shape[:2] == want
        lab = f'{out.shape[1]} x {out.shape[0]}'
        if not size_ok:
            lab += f'   want {want[1]} x {want[0]}'
            same = False
        else:                       # 크기가 맞아도 내용이 맞는지 따로 봐야 한다
            d = np.abs(out.astype(np.int16) - good.astype(np.int16)).mean()
            same = d < 0.5
            lab += '' if i == 1 else f'   off by {d:.1f} levels'
        ok = size_ok and same
        ax.imshow(out)
        ax.set_title(f'{i}.  {name}', fontsize=20, pad=10, loc='left',
                     color='#222' if ok else C_EDGE)
        ax.set_xlabel(lab, fontsize=17, color='#222' if ok else C_EDGE)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_box_aspect(1)
        for sp in ax.spines.values():
            sp.set(color='#222' if ok else C_EDGE, linewidth=2.4)
    plt.show()


def try_run(img, net, cfg):
    """Run, and if the GPU runs out of memory say so instead of dying."""
    try:
        if cfg.device.startswith('cuda'):
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(cfg.device)
        t0 = time.time()
        out = sr_tiled(img, net, cfg)
        p = plan(*img.shape[:2], cfg)
        print(f'ok    {p["px_per_call"] / 1e6:.2f} M px per call   '
              f'{time.time() - t0:.2f} s   '
              f'{torch.cuda.max_memory_allocated(cfg.device) / 2 ** 30:.2f} GiB')
        return out
    except torch.cuda.OutOfMemoryError as e:
        torch.cuda.empty_cache()
        p = plan(*img.shape[:2], cfg)
        print(f'OOM   {p["px_per_call"] / 1e6:.2f} M px per call')
        print(f'      {str(e).splitlines()[0]}')
        return None


def show_result(lr, sr, cfg, center=None, size=120):
    """Input, output, and the same patch in both at the same screen size."""
    h, w = lr.shape[:2]
    cy, cx = center or (h // 2, w // 2)
    y = int(np.clip(cy - size // 2, 0, h - size)); x = int(np.clip(cx - size // 2, 0, w - size))
    s = cfg.scale
    fig, axes = plt.subplots(1, 4, figsize=(18.0, 5.2), layout='constrained')
    axes[0].imshow(lr); axes[0].add_patch(Rectangle((x, y), size, size, fill=False,
                                                    ec=C_EDGE, lw=2.2))
    axes[1].imshow(sr); axes[1].add_patch(Rectangle((x * s, y * s), size * s, size * s,
                                                    fill=False, ec=C_EDGE, lw=2.2))
    axes[2].imshow(lr[y:y + size, x:x + size], interpolation='nearest')
    axes[3].imshow(sr[y * s:(y + size) * s, x * s:(x + size) * s], interpolation='nearest')
    for ax, t in zip(axes, (f'1.  input   {w} x {h}',
                            f'2.  output   {w * s} x {h * s}',
                            '3.  input, zoomed', f'4.  output, zoomed')):
        ax.set_xticks([]); ax.set_yticks([]); ax.set_box_aspect(1)
        ax.set_title(t, fontsize=19, pad=10, loc='left')
    plt.show()


def memory_law(img, net, cfg, tiles=(128, 256, 512), batches=(1, 4, 8, 16)):
    """Peak memory against pixels per call, reached two different ways."""
    rows = []
    for t in tiles:
        rows.append(('tile', t, replace(cfg, tile=t, batch=4)))
    for b in batches:
        rows.append(('batch', b, replace(cfg, tile=256, batch=b)))
    out = []
    for kind, val, c in rows:
        try:
            torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(c.device)
            sr_tiled(img, net, c)
            out.append(dict(kind=kind, val=val,
                            mpx=plan(*img.shape[:2], c)['px_per_call'] / 1e6,
                            vram=torch.cuda.max_memory_allocated(c.device) / 2 ** 30))
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
    return out


def show_memory_law(rows):
    """Both knobs land on the same line: memory follows pixels per call."""
    fig, ax = plt.subplots(figsize=(8.6, 6.0), layout='constrained')
    for kind, col, mk in (('tile', C_KEEP, 'o'), ('batch', C_READ, 's')):
        g = [r for r in rows if r['kind'] == kind]
        ax.plot([r['mpx'] for r in g], [r['vram'] for r in g], mk, ms=13,
                color=col, label=f'{kind} varied', ls='none')
        for r in g:
            ax.annotate(str(r['val']), (r['mpx'], r['vram']), fontsize=13,
                        textcoords='offset points', xytext=(9, -4), color=col)
    x = np.array([r['mpx'] for r in rows]); y = np.array([r['vram'] for r in rows])
    k = float((x * y).sum() / (x * x).sum())
    xs = np.linspace(0, x.max() * 1.1, 2)
    ax.plot(xs, k * xs, '--', color='#666', lw=1.8, zorder=0,
            label=f'{k:.2f} GiB per M px')
    ax.set_xlabel('M pixels per forward pass', fontsize=18)
    ax.set_ylabel('peak VRAM  (GiB)', fontsize=18)
    ax.tick_params(labelsize=15); ax.grid(alpha=0.25)
    ax.legend(fontsize=15, frameon=False, loc='upper left')
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    plt.show()
