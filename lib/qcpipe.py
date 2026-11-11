"""One scene, end to end: prepare, infer in tiles, write in windows, check, score, report.

    scan          header + the 8-bit conversion, matched to a reference histogram
    run           read a block, convert, infer in tiles, convert back, write that block
    qc            header, radiometry and seam checks, each one pass / CHECK
    iqa_scan      no-reference scores and re-downscale consistency on a few crops
    write_report  everything above as one self-contained HTML file plus a JSON

Reading and writing go through rasterio windows, so only one block is in memory at a time.
"""
import base64
import io
import json
import os
import time
from datetime import datetime, timezone

import cv2
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import torch
from matplotlib.patches import Rectangle
from rasterio.enums import Resampling
from rasterio.transform import Affine
from rasterio.windows import Window

import iqa

__all__ = ['scan', 'show_scan', 'blocks', 'show_blocks', 'run', 'qc', 'show_qc',
           'iqa_scan', 'show_iqa', 'write_report', 'break_output', 'np', 'plt']

C_IN, C_OUT, C_OK, C_BAD = '#2b6cb0', '#f0a63a', '#2e8b57', '#c0392b'
BAND_RGB = ('#c0392b', '#2e8b57', '#2b6cb0')
NAMES = ('red', 'green', 'blue', 'nir')


# --------------------------------------------------------------- 1. prepare
def scan(path, reference, bands=(1, 2, 3), size=1024):
    """Header plus the conversion to 8 bits.  Each band is histogram-matched to
    `reference` (quantiles of the model's 8-bit training inputs): a value at the p-th
    percentile of the scene becomes the p-th percentile of the reference.  The scene
    quantiles come from a read decimated to about `size` px a side."""
    r = np.load(reference)
    q, ref = r['q'], r['ref']
    with rasterio.open(path) as d:
        k = max(1, -(-max(d.width, d.height) // size))
        ov = d.read(out_shape=(d.count, d.height // k, d.width // k),
                    resampling=Resampling.nearest)
        m = dict(path=path, width=d.width, height=d.height, count=d.count,
                 dtype=d.dtypes[0], nodata=d.nodata)
    empty = (ov == 0).all(0)
    bits = int(np.ceil(np.log2(ov.max() + 1)))
    grid = np.arange(2 ** bits, dtype=np.float32)
    fwd, back, src_q = [], [], []
    for i, b in enumerate(bands):
        sq = np.percentile(ov[b - 1][~empty], q).astype(np.float64)
        sq += np.arange(len(sq)) * 1e-6               # strictly increasing for interp
        f = np.interp(grid, sq, ref[i]).round().clip(0, 255).astype(np.uint8)
        # back: each 8-bit level goes to the middle of the scene values that map to it
        lv = ref[i].round()
        bk = np.array([sq[lv == u].mean() if (lv == u).any() else np.nan
                       for u in range(256)])
        ok = ~np.isnan(bk)
        bk = np.interp(np.arange(256), np.flatnonzero(ok), bk[ok])
        fwd.append(f); back.append(bk); src_q.append(sq)
    m.update(bands=tuple(bands), decimation=k, bits=bits, overview=ov,
             fwd=np.stack(fwd), back=np.stack(back), q=q, ref=ref, src_q=np.stack(src_q))
    return m


def _to8(a, s):
    """(band, h, w) raw -> (h, w, 3) uint8, through the matched lookup tables."""
    a = np.minimum(a, s['fwd'].shape[1] - 1)
    return np.stack([s['fwd'][i][a[i]] for i in range(len(a))], -1)


def _back(x8, s, dtype):
    """(h, w, 3) uint8 -> (band, h, w) in the original range.  Floor at 1 so a real
    pixel never reads as nodata."""
    x = np.stack([s['back'][i][x8[..., i]] for i in range(x8.shape[-1])])
    return np.clip(x.round(), 1, np.iinfo(dtype).max).astype(dtype)


def _rgb(a, s):
    return _to8(a[[b - 1 for b in s['bands']]], s)


def show_scan(s, bins=64):
    """The scene after conversion, and its 8-bit histogram against the reference."""
    ov = s['overview']
    x8 = _rgb(ov, s)
    fig, ax = plt.subplots(1, 2, figsize=(16.5, 6.6), layout='constrained',
                           gridspec_kw=dict(width_ratios=[1, 1.35]))
    ax[0].imshow(x8)
    ax[0].set_xticks([]); ax[0].set_yticks([])
    ax[0].set_title('1.  scene, converted to 8 bits', fontsize=21, pad=12, loc='left')
    ax[0].set_xlabel(f'{s["width"]} x {s["height"]} px   {s["count"]} bands   '
                     f'{s["dtype"]}', fontsize=17)
    edges = np.linspace(0, 256, bins + 1)
    for i, b in enumerate(s['bands']):
        h = np.histogram(x8[..., i], edges)[0]
        r = np.histogram(s['ref'][i], edges)[0]
        ax[1].step(edges[:-1], h / h.sum(), where='post', lw=2.4, color=BAND_RGB[i],
                   label=NAMES[b - 1])
        ax[1].step(edges[:-1], r / r.sum(), where='post', lw=1.4, color=BAND_RGB[i], ls='--')
    ax[1].plot([], [], color='k', lw=2.4, label='scene')
    ax[1].plot([], [], color='k', lw=1.4, ls='--', label='reference')
    ax[1].set_yticks([]); ax[1].set_xlim(0, 255)
    ax[1].tick_params(labelsize=14); ax[1].legend(fontsize=14, frameon=False, ncol=2)
    ax[1].set_title('2.  8-bit histogram, matched to the reference', fontsize=21, pad=12,
                    loc='left')
    ax[1].set_xlabel('pixel value     0 - 255', fontsize=17)
    for sp in ('top', 'right', 'left'):
        ax[1].spines[sp].set_visible(False)
    plt.show()


# --------------------------------------------------------------- 2. run
def blocks(s, block):
    """Block origins and sizes, in input pixels."""
    return [(y, x, min(block, s['height'] - y), min(block, s['width'] - x))
            for y in range(0, s['height'], block) for x in range(0, s['width'], block)]


def show_blocks(s, block, cfg, which=None):
    """Solid blue is what each block writes.  Dashed orange is what one block reads."""
    k = s['decimation']
    bl = blocks(s, block)
    fig, ax = plt.subplots(figsize=(8.5, 8.5 * s['height'] / s['width']),
                           layout='constrained')
    ax.imshow(_rgb(s['overview'], s), alpha=0.75)
    for y, x, h, w in bl:
        ax.add_patch(Rectangle((x / k, y / k), w / k, h / k, fc='none', ec=C_IN, lw=2))
    y, x, h, w = bl[which if which is not None else len(bl) // 2]
    m = cfg.overlap
    ax.add_patch(Rectangle(((x - m) / k, (y - m) / k), (w + 2 * m) / k, (h + 2 * m) / k,
                           fc='none', ec=C_OUT, lw=2.6, ls='--'))
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f'{len(bl)} blocks of {block} px, margin {m} px',
                 fontsize=21, pad=12, loc='left')
    plt.show()


def _profile(src, s, cfg):
    t = src.transform
    p = dict(driver='GTiff', width=s['width'] * cfg.scale, height=s['height'] * cfg.scale,
             count=len(s['bands']), dtype=s['dtype'], crs=src.crs,
             transform=t * Affine.scale(1 / cfg.scale), nodata=src.nodata,
             tiled=True, blockxsize=256, blockysize=256, compress='deflate',
             predictor=2, BIGTIFF='IF_SAFER', photometric='RGB')
    return p


def _sr_block(a, net, cfg, value_range):
    """a: (h + 2m, w + 2m, 3) uint8 — one block with its margin already around it.
    Tiles start at the block origin, so with `block` a multiple of `tile` every tile
    boundary in the scene falls on a multiple of `tile`."""
    t, m, sc = cfg.tile, cfg.overlap, cfg.scale
    h, w = a.shape[0] - 2 * m, a.shape[1] - 2 * m
    ny, nx = -(-h // t), -(-w // t)
    a = np.pad(a, ((0, ny * t - h), (0, nx * t - w), (0, 0)), mode='reflect')
    out = np.empty((ny * t * sc, nx * t * sc, 3), np.uint8)
    keep, edge = t * sc, m * sc
    pos = [(iy, ix) for iy in range(ny) for ix in range(nx)]
    for i in range(0, len(pos), cfg.batch):
        chunk = pos[i:i + cfg.batch]
        x = np.stack([a[iy * t:iy * t + t + 2 * m, ix * t:ix * t + t + 2 * m]
                      for iy, ix in chunk])
        x = torch.from_numpy(x).permute(0, 3, 1, 2).float().to(cfg.device) * (value_range / 255)
        with torch.no_grad(), torch.autocast(cfg.device.split(':')[0], dtype=torch.float16,
                                             enabled=cfg.precision == 'fp16'):
            y = net(x)
        y = (y.float() * (255 / value_range)).clamp(0, 255).round().byte()
        y = y.permute(0, 2, 3, 1).cpu().numpy()
        for (iy, ix), tile in zip(chunk, y):
            out[iy * keep:(iy + 1) * keep, ix * keep:(ix + 1) * keep] = \
                tile[edge:edge + keep, edge:edge + keep]
    return out[:h * sc, :w * sc]


def run(src_path, dst_path, net, cfg, s, block=768, model='HAT x3', value_range=1.0):
    """Block by block: read with a margin, convert, infer in tiles, convert back,
    write.  Only one block is ever in memory.

    value_range   what the network expects for white: 1.0 (HAT, SwinIR) or 255 (EDSR)
    """
    sc, m = cfg.scale, cfg.overlap
    bl = blocks(s, block)
    dt = np.dtype(s['dtype'])
    if cfg.device.startswith('cuda'):
        torch.cuda.reset_peak_memory_stats(cfg.device)
    t0, peak = time.time(), 0
    with rasterio.open(src_path) as src, \
            rasterio.open(dst_path, 'w', **_profile(src, s, cfg)) as dst:
        for y, x, h, w in bl:
            # read the margin where the scene has it, reflect where it does not
            y0, x0 = max(y - m, 0), max(x - m, 0)
            y1, x1 = min(y + h + m, s['height']), min(x + w + m, s['width'])
            a = src.read(s['bands'], window=Window(x0, y0, x1 - x0, y1 - y0))
            a8 = np.pad(_to8(a, s), ((m - (y - y0), y + h + m - y1),
                                     (m - (x - x0), x + w + m - x1), (0, 0)), mode='reflect')
            sr = _sr_block(a8, net, cfg, value_range)
            out = _back(sr, s, dt)
            # input nodata stays nodata
            empty = (a == 0).all(0)[y - y0:y - y0 + h, x - x0:x - x0 + w]
            out[:, np.repeat(np.repeat(empty, sc, 0), sc, 1)] = 0
            dst.write(out, window=Window(x * sc, y * sc, w * sc, h * sc))
            peak = max(peak, a.nbytes + a8.nbytes + sr.nbytes + out.nbytes)
        dst.update_tags(SR_MODEL=model, SR_SCALE=sc, SR_SOURCE=os.path.basename(src_path),
                        SR_INPUT='8-bit, histogram matched to the training inputs',
                        SR_TILE=cfg.tile, SR_OVERLAP=cfg.overlap, SR_BLOCK=block,
                        SR_DATE=datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'))
        for i, b in enumerate(s['bands'], start=1):
            dst.set_band_description(i, NAMES[b - 1])
    vram = (torch.cuda.max_memory_allocated(cfg.device) / 2 ** 30
            if cfg.device.startswith('cuda') else 0.0)
    st = dict(blocks=len(bl), block=block, seconds=round(time.time() - t0, 1),
              gpu_gib=round(vram, 2), ram_block_mb=round(peak / 2 ** 20, 1), out=dst_path,
              out_mb=round(os.path.getsize(dst_path) / 2 ** 20, 1),
              tile=cfg.tile, overlap=cfg.overlap, batch=cfg.batch,
              precision=cfg.precision, device=cfg.device, scale=sc, model=model)
    print(f'{st["blocks"]} blocks   {st["seconds"]} s   GPU {st["gpu_gib"]} GiB   '
          f'RAM per block {st["ram_block_mb"]} MB   file {st["out_mb"]} MB')
    return st


# --------------------------------------------------------------- 3. QC
def _hist(path, bands, rows=1024):
    """Exact per-band histograms, accumulated strip by strip."""
    with rasterio.open(path) as d:
        top = np.iinfo(d.dtypes[0]).max + 1
        H = np.zeros((len(bands), top), np.int64)
        for y in range(0, d.height, rows):
            a = d.read(bands, window=Window(0, y, d.width, min(rows, d.height - y)))
            for i in range(len(bands)):
                H[i] += np.bincount(a[i].ravel(), minlength=top)
    return H


def _moments(h):
    v = np.arange(len(h), dtype=np.float64)
    n = h.sum()
    mu = (h * v).sum() / n
    nz = np.flatnonzero(h)
    return dict(mean=mu, std=np.sqrt((h * (v - mu) ** 2).sum() / n),
                min=int(nz[0]), max=int(nz[-1]), n=int(n))


def _seam_ratio(path, period, scale, rows=12):
    """Mean step across every `period`-th row and column, over the mean step a few
    rows away.  A visible seam pushes it above 1.  The rows compared sit a multiple
    of `scale` away, because the upsampler repeats its pattern every `scale` rows."""
    near = [rows - 1 + k * scale for k in (-3, -2, -1, 1, 2, 3)]
    with rasterio.open(path) as d:
        W, H = d.width, d.height
        r = []
        for axis, n in ((0, H), (1, W)):
            at, base = [], []
            for p in range(period, n - rows, period):
                win = (Window(0, p - rows, W, 2 * rows) if axis == 0
                       else Window(p - rows, 0, 2 * rows, H))
                a = d.read(1, window=win).astype(np.float32)
                g = np.abs(np.diff(a, axis=axis)).mean(axis=1 - axis)
                at.append(g[rows - 1]); base.append(g[near].mean())
            if at:
                r.append(np.mean(at) / np.mean(base))
    return float(max(r)) if r else 1.0


def qc(src_path, dst_path, s, cfg, block=768):
    """Every check as a row: group, name, expected, got, ok."""
    rows = []

    def add(g, name, exp, got, ok):
        rows.append(dict(group=g, check=name, expected=str(exp), got=str(got), ok=bool(ok)))

    with rasterio.open(src_path) as a, rasterio.open(dst_path) as b:
        sc, nb = cfg.scale, len(s['bands'])
        add('header', 'driver', 'GTiff', b.driver, b.driver == 'GTiff')
        add('header', 'width', f'{a.width} x {sc}', b.width, b.width == a.width * sc)
        add('header', 'height', f'{a.height} x {sc}', b.height, b.height == a.height * sc)
        add('header', 'bands', nb, b.count, b.count == nb)
        add('header', 'dtype', a.dtypes[0], b.dtypes[0], b.dtypes[0] == a.dtypes[0])
        add('header', 'nodata', a.nodata, b.nodata, b.nodata == a.nodata)
        add('header', 'tiled', True, b.profile.get('tiled', False), b.profile.get('tiled', False))
        add('header', 'compression', 'any', b.compression.value if b.compression else None,
            b.compression is not None)
        tags = b.tags()
        need = ('SR_MODEL', 'SR_SCALE', 'SR_SOURCE', 'SR_INPUT')
        miss = [k for k in need if k not in tags]
        add('header', 'processing tags', ', '.join(need), 'missing ' + ', '.join(miss)
            if miss else 'all present', not miss)

    Ha, Hb = _hist(src_path, s['bands']), _hist(dst_path, list(range(1, nb + 1)))
    top = 2 ** s['bits'] - 1
    for i, band in enumerate(s['bands']):
        nm = NAMES[band - 1]
        ma, mb = _moments(Ha[i]), _moments(Hb[i])
        dm = 100 * (mb['mean'] / ma['mean'] - 1)
        add('radiometry', f'{nm} mean', f'{ma["mean"]:.1f}  (within 2 %)',
            f'{mb["mean"]:.1f}  ({dm:+.2f} %)', abs(dm) <= 2)
        rs = mb['std'] / ma['std']
        add('radiometry', f'{nm} std', f'{ma["std"]:.1f}  (x 0.9 - 1.3)',
            f'{mb["std"]:.1f}  (x {rs:.3f})', 0.9 <= rs <= 1.3)
        add('radiometry', f'{nm} range', f'{ma["min"]} - {ma["max"]}  (1 - {top})',
            f'{mb["min"]} - {mb["max"]}', mb['min'] >= 1 and mb['max'] <= top)
        # shape: 32 bins that each hold 1/32 of the input; the output should fill them evenly
        ca = np.cumsum(Ha[i]) / ma['n']
        edges = np.searchsorted(ca, np.linspace(0, 1, 33)[1:-1])
        pb = np.diff(np.concatenate([[0], np.cumsum(Hb[i])[edges], [mb['n']]])) / mb['n']
        pa = np.diff(np.concatenate([[0], np.cumsum(Ha[i])[edges], [ma['n']]])) / ma['n']
        bc = float(np.sqrt(pa * pb).sum())          # Bhattacharyya coefficient, 1 = same shape
        add('radiometry', f'{nm} histogram shape', '>= 0.98', f'{bc:.4f}', bc >= 0.98)
    holes_a = int(Ha[:, 0].min()) * cfg.scale ** 2
    holes_b = int(Hb[:, 0].max())
    add('radiometry', 'nodata pixels', f'<= {holes_a}', holes_b, holes_b <= holes_a)

    for nm, period in (('tile seams', cfg.tile * cfg.scale), ('block seams', block * cfg.scale)):
        r = _seam_ratio(dst_path, period, cfg.scale)
        add('seams', nm, '<= 1.15', f'{r:.3f}', r <= 1.15)
    return rows


def show_qc(rows, only_bad=False):
    """The checks as a table: green passes, red needs a look."""
    import pandas as pd
    df = pd.DataFrame(rows)
    df['ok'] = df['ok'].map({True: 'pass', False: 'CHECK'})
    n = (df['ok'] == 'CHECK').sum()
    print(f'{len(df)} checks   {len(df) - n} pass   {n} CHECK')
    if only_bad:
        df = df[df['ok'] == 'CHECK']
    return (df.style.hide(axis='index')
            .map(lambda v: f'color: white; background: {C_OK if v == "pass" else C_BAD}',
                 subset=['ok'])
            .set_properties(**{'text-align': 'left'}))


# --------------------------------------------------------------- 4. IQA
def _pick(s, n_lr):
    """Busy, typical and flat places, chosen by local contrast on the overview."""
    ov = s['overview'][[b - 1 for b in s['bands']]].mean(0)
    k = s['decimation']
    c = max(1, n_lr // k)
    ny, nx = ov.shape[0] // c, ov.shape[1] // c
    sd = np.array([[ov[j * c:(j + 1) * c, i * c:(i + 1) * c].std()
                    for i in range(nx)] for j in range(ny)])
    order = np.argsort(sd.ravel())
    out = []
    for name, q in (('busy', 0.95), ('typical', 0.5), ('flat', 0.1)):
        j, i = divmod(int(order[int(q * (len(order) - 1))]), nx)
        out.append((name, j * c * k, i * c * k))
    return out


def iqa_scan(src_path, dst_path, s, cfg, size=200, iqa_dir=None):
    """Three crops.  For each, bicubic and SR scored with NIQE / BRISQUE / PIQE, and
    PSNR / SSIM of the result shrunk back to the input grid against the input."""
    iqa.load_models(iqa_dir)
    sc = cfg.scale
    out = []
    with rasterio.open(src_path) as a, rasterio.open(dst_path) as b:
        for name, y, x in _pick(s, size):
            lr = _to8(a.read(s['bands'], window=Window(x, y, size, size)), s)
            sr = _to8(b.read(window=Window(x * sc, y * sc, size * sc, size * sc)), s)
            bic = cv2.resize(lr, (size * sc, size * sc), interpolation=cv2.INTER_CUBIC)
            row = dict(crop=name, y=int(y), x=int(x), lr=lr, imgs={})
            for k, im in (('bicubic', bic), ('SR', sr)):
                back = cv2.resize(im, (size, size), interpolation=cv2.INTER_AREA)
                row['imgs'][k] = im
                row[k] = dict(niqe=iqa.niqe(im), brisque=iqa.brisque(im),
                              piqe=iqa.piqe(im), psnr=iqa.psnr(back, lr),
                              ssim=iqa.ssim(back, lr))
            out.append(row)
    return out


def show_iqa(res, zoom=240):
    """Each crop: the input, bicubic and SR, with the scores of the last two."""
    for r in res:
        z = zoom
        lr = cv2.resize(r['lr'], None, fx=r['imgs']['SR'].shape[0] // r['lr'].shape[0],
                        fy=r['imgs']['SR'].shape[0] // r['lr'].shape[0],
                        interpolation=cv2.INTER_NEAREST)
        items = [('input', lr[:z, :z], None)] + [(k, r['imgs'][k][:z, :z], r[k])
                                                for k in ('bicubic', 'SR')]
        fig, ax = plt.subplots(1, 3, figsize=(15, 5.6), layout='constrained')
        for a, (k, im, v) in zip(ax, items):
            a.imshow(im, interpolation='nearest'); a.set_xticks([]); a.set_yticks([])
            a.set_title(f'{k}  —  {r["crop"]}' if v is None else k, fontsize=20, pad=10,
                        loc='left')
            if v:
                a.set_xlabel(f'NIQE {v["niqe"]:.2f}   BRISQUE {v["brisque"]:.1f}   '
                             f'PIQE {v["piqe"]:.1f}\nback to input   PSNR {v["psnr"]:.2f} dB'
                             f'   SSIM {v["ssim"]:.3f}', fontsize=15, linespacing=1.6)
        plt.show()


def iqa_rows(res, min_psnr=30.0):
    """The consistency scores become checks too."""
    return [dict(group='iqa', check=f'{r["crop"]} consistency',
                 expected=f'PSNR >= {min_psnr:g} dB',
                 got=f'{r["SR"]["psnr"]:.2f} dB  SSIM {r["SR"]["ssim"]:.3f}',
                 ok=r['SR']['psnr'] >= min_psnr) for r in res]


__all__ += ['iqa_rows']


# --------------------------------------------------------------- 5. report
def _png(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=80, bbox_inches='tight')
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def _fig_overview(s, dst_path):
    with rasterio.open(dst_path) as d:
        k = max(1, -(-max(d.width, d.height) // 1024))
        ov = d.read(out_shape=(d.count, d.height // k, d.width // k),
                    resampling=Resampling.nearest)
    fig, ax = plt.subplots(1, 2, figsize=(12, 6), layout='constrained')
    for a, im, t in ((ax[0], _rgb(s['overview'], s), 'input'),
                     (ax[1], _to8(ov, s), 'output')):
        a.imshow(im); a.set_xticks([]); a.set_yticks([]); a.set_title(t, loc='left')
    return _png(fig)


def _fig_crops(res, zoom=240):
    fig, ax = plt.subplots(len(res), 3, figsize=(9, 3.1 * len(res)), layout='constrained',
                           squeeze=False)
    for j, r in enumerate(res):
        f = r['imgs']['SR'].shape[0] // r['lr'].shape[0]
        lr = cv2.resize(r['lr'], None, fx=f, fy=f, interpolation=cv2.INTER_NEAREST)
        for i, (k, im) in enumerate((('input', lr), ('bicubic', r['imgs']['bicubic']),
                                     ('SR', r['imgs']['SR']))):
            a = ax[j][i]
            a.imshow(im[:zoom, :zoom], interpolation='nearest')
            a.set_xticks([]); a.set_yticks([])
            if j == 0:
                a.set_title(k, loc='left')
            if i == 0:
                a.set_ylabel(r['crop'])
    return _png(fig)


def _fig_hist(src_path, dst_path, s):
    Ha = _hist(src_path, s['bands'])
    Hb = _hist(dst_path, list(range(1, len(s['bands']) + 1)))
    top = 2 ** s['bits']
    fig, ax = plt.subplots(1, len(s['bands']), figsize=(12, 3.4), layout='constrained')
    for i, band in enumerate(s['bands']):
        for h, c, nm in ((Ha[i], C_IN, 'input'), (Hb[i], C_OUT, 'output')):
            p = h[:top].reshape(-1, 16).sum(1) / h.sum()
            ax[i].step(np.arange(len(p)) * 16, p, color=c, lw=1.6, label=nm)
        ax[i].set_yscale('log'); ax[i].set_yticks([]); ax[i].set_title(NAMES[band - 1], loc='left')
        ax[i].legend(frameon=False)
    return _png(fig)


_CSS = """
.qcr{font-family:system-ui,-apple-system,Segoe UI,sans-serif;color:#1f2933;max-width:980px}
.qcr h1{font-size:24px;margin:0 0 4px}.qcr h2{font-size:18px;margin:26px 0 8px}
.qcr .sub{color:#52606d;margin-bottom:14px}
.qcr .badge{display:inline-block;padding:4px 14px;border-radius:4px;color:#fff;font-weight:600}
.qcr table{border-collapse:collapse;width:100%;font-size:13px}
.qcr th,.qcr td{border-bottom:1px solid #e4e7eb;padding:5px 8px;text-align:left}
.qcr th{background:#f5f7fa}.qcr td.ok{color:#fff;text-align:center;width:60px}
.qcr img{max-width:100%}.qcr code{background:#f5f7fa;padding:1px 4px}
"""


def _table(rows, cols):
    h = ''.join(f'<th>{c}</th>' for c in cols)
    body = ''
    for r in rows:
        tds = ''
        for c in cols:
            v = r[c]
            if c == 'ok':
                tds += (f'<td class="ok" style="background:{C_OK if v else C_BAD}">'
                        f'{"pass" if v else "CHECK"}</td>')
            else:
                tds += f'<td>{v}</td>'
        body += f'<tr>{tds}</tr>'
    return f'<table><tr>{h}</tr>{body}</table>'


def write_report(out_html, src_path, dst_path, s, st, checks, res):
    """One HTML file with every number and picture embedded, and a JSON beside it."""
    bad = [c for c in checks if not c['ok']]
    verdict = 'PASS' if not bad else f'CHECK  ({len(bad)} of {len(checks)})'
    when = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')
    run_rows = [dict(item=k, value=v) for k, v in (
        ('input', os.path.basename(src_path)),
        ('input size', f'{s["width"]} x {s["height"]} px, {s["count"]} bands, {s["dtype"]}'),
        ('bands used', ', '.join(NAMES[b - 1] for b in s['bands'])),
        ('conversion', '8-bit, histogram matched to the training inputs'),
        ('model', f'{st["model"]}   scale {st["scale"]}'),
        ('tiling', f'tile {st["tile"]}  overlap {st["overlap"]}  batch {st["batch"]}  '
                   f'{st["precision"]}  {st["device"]}'),
        ('blocks', f'{st["blocks"]} of {st["block"]} px'),
        ('time', f'{st["seconds"]} s'), ('GPU peak', f'{st["gpu_gib"]} GiB'),
        ('RAM per block', f'{st["ram_block_mb"]} MB'),
        ('output', f'{os.path.basename(dst_path)}  {st["out_mb"]} MB'))]
    iqa_tab = [dict(crop=r['crop'], method=k, **{m: f'{r[k][m]:.3f}' for m in
                                                 ('niqe', 'brisque', 'piqe', 'psnr', 'ssim')})
               for r in res for k in ('bicubic', 'SR')]
    groups = ''
    for g in dict.fromkeys(c['group'] for c in checks):
        groups += f'<h2>{g}</h2>' + _table([c for c in checks if c['group'] == g],
                                            ['check', 'expected', 'got', 'ok'])
    html = f"""<div class="qcr"><style>{_CSS}</style>
<h1>SR QC report &mdash; {os.path.basename(src_path)}</h1>
<div class="sub">{when}</div>
<span class="badge" style="background:{C_OK if not bad else C_BAD}">{verdict}</span>
<h2>run</h2>{_table(run_rows, ['item', 'value'])}
<h2>overview</h2><img src="data:image/png;base64,{_fig_overview(s, dst_path)}">
{groups}
<h2>histograms</h2><img src="data:image/png;base64,{_fig_hist(src_path, dst_path, s)}">
<h2>image quality</h2>
<p>NIQE, BRISQUE, PIQE: lower is better, relative to bicubic only.
PSNR, SSIM: the result shrunk back to the input grid, against the input.</p>
{_table(iqa_tab, ['crop', 'method', 'niqe', 'brisque', 'piqe', 'psnr', 'ssim'])}
<img src="data:image/png;base64,{_fig_crops(res)}">
</div>"""
    with open(out_html, 'w') as f:
        f.write(f'<!doctype html><meta charset="utf-8"><title>SR QC report</title>{html}')
    out_json = os.path.splitext(out_html)[0] + '.json'
    with open(out_json, 'w') as f:
        json.dump(dict(verdict='PASS' if not bad else 'CHECK', created=when,
                       input=src_path, output=dst_path, run=st,
                       checks=checks,
                       iqa=[{k: r[k] for k in ('crop', 'y', 'x', 'bicubic', 'SR')} for r in res]),
                  f, indent=1, default=str)
    print(f'{verdict}   ->  {out_html}, {out_json}')
    return html


# --------------------------------------------------------------- a broken output
def break_output(dst_path, out_path, s):
    """Copy the result as it would be if the network's 8-bit output had been written
    without converting back."""
    with rasterio.open(dst_path) as d:
        with rasterio.open(out_path, 'w', **d.profile) as o:
            for _, w in d.block_windows(1):
                o.write(np.transpose(_to8(d.read(window=w), s), (2, 0, 1)).astype(d.dtypes[0]),
                        window=w)
            o.update_tags(**d.tags())
    return out_path
