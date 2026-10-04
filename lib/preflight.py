"""Checks to run before launching an SR job, so it does not fail an hour in.

    gpu_state        what card is here, and how much of it is free
    budget           measure cost per pixel, then say what will fit
    to_8bit          put a 12-bit scene into the range the model was trained on
    show_conversion  what happens when that step is skipped
    save_sr          write the result back with the input's georeferencing
"""
import numpy as np
import matplotlib.pyplot as plt
import rasterio
import torch

from infer import Config, plan, replace, sr_tiled

__all__ = ['gpu_state', 'measure', 'budget', 'show_budget', 'to_8bit',
           'show_conversion', 'save_sr', 'np', 'plt', 'torch']

C_OK, C_BAD, C_IN = '#2e8b57', '#c0392b', '#2b6cb0'


# --------------------------------------------------------------- Q1
def gpu_state(device=None):
    """Name, memory, and whether fp16 will actually be fast here."""
    device = device or ('cuda:0' if torch.cuda.is_available() else 'cpu')
    if not str(device).startswith('cuda') or not torch.cuda.is_available():
        print('no GPU in this runtime — everything below would run on the CPU, and')
        print('EDSR on a CPU takes minutes per scene rather than seconds.')
        print('Runtime > Change runtime type > T4 GPU, then run the cells again.')
        return
    i = torch.device(device).index or 0
    p = torch.cuda.get_device_properties(i)
    free, total = torch.cuda.mem_get_info(i)
    print(f'device     {p.name}')
    print(f'memory     {total / 2 ** 30:.1f} GiB total, {free / 2 ** 30:.1f} GiB free')
    print(f'compute    {p.major}.{p.minor}'
          f'   tensor cores {"yes" if p.major >= 7 else "no"}')
    print(f'torch      {torch.__version__}')


def _no_gpu(cfg):
    if cfg.device.startswith('cuda'):
        return False
    print('no GPU in this runtime — there is no memory limit to plan around.')
    print('Runtime > Change runtime type > T4 GPU, then run the cells again.')
    return True


def measure(img, net, cfg, tiles=(128, 256)):
    """How many GiB one million pixels per forward pass costs on this card.
    Two small runs are enough — the relation is a straight line through zero."""
    pts = []
    for t in tiles:
        c = replace(cfg, tile=t)
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(c.device)
        sr_tiled(img, net, c)
        pts.append((plan(*img.shape[:2], c)['px_per_call'] / 1e6,
                    torch.cuda.max_memory_allocated(c.device) / 2 ** 30))
    x = np.array([p[0] for p in pts]); y = np.array([p[1] for p in pts])
    return float((x * y).sum() / (x * x).sum()), pts


def budget(img, net, cfg, margin=0.85):
    """Turn that number into the configs that will and will not fit."""
    if _no_gpu(cfg):
        return None, None
    per, _ = measure(img, net, cfg)
    i = torch.device(cfg.device).index or 0
    free = torch.cuda.mem_get_info(i)[0] / 2 ** 30
    cap = free * margin / per
    print(f'cost       {per:.2f} GiB per M px in one forward pass')
    print(f'free       {free:.1f} GiB,  usable {margin:.0%}  ->  {cap:.2f} M px per call')
    print()
    print(f'{"tile":>6} {"reads":>7} {"max batch":>10} {"M px":>8}')
    for t in (128, 256, 512, 1024):
        c = replace(cfg, tile=t)
        b = int(cap * 1e6 // c.read ** 2)
        print(f'{t:6d} {c.read:7d} {b if b else "-- none --":>10} '
              f'{b * c.read ** 2 / 1e6:8.2f}')
    return per, cap


def show_budget(img, net, cfg, margin=0.85):
    """The measured line, and where this card stops."""
    if _no_gpu(cfg):
        return
    per, pts = measure(img, net, cfg)
    i = torch.device(cfg.device).index or 0
    free, total = (v / 2 ** 30 for v in torch.cuda.mem_get_info(i))
    cap = free * margin / per
    x = np.linspace(0, cap * 1.35, 2)

    fig, ax = plt.subplots(figsize=(9.2, 5.8), layout='constrained')
    ax.plot(x, per * x, '--', color='#666', lw=2.0, label=f'{per:.2f} GiB per M px')
    ax.plot([p[0] for p in pts], [p[1] for p in pts], 'o', ms=13, color=C_IN,
            label='measured')
    ax.axhline(total, color=C_BAD, lw=2.2)
    ax.text(0, total, f' {total:.1f} GiB on this card', color=C_BAD, fontsize=16,
            va='bottom')
    ax.axvline(cap, color=C_OK, lw=2.2)
    ax.text(cap, 0, f' {cap:.1f} M px ', color=C_OK, fontsize=16, va='bottom',
            ha='left')
    ax.set_xlim(0, x[-1]); ax.set_ylim(0, total * 1.25)
    ax.set_xlabel('M pixels per forward pass', fontsize=18)
    ax.set_ylabel('peak VRAM  (GiB)', fontsize=18)
    ax.tick_params(labelsize=15); ax.grid(alpha=0.25)
    ax.legend(fontsize=15, frameon=False, loc='upper left')
    ax.set_title('1.  what fits', fontsize=21, pad=12, loc='left')
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    plt.show()


# --------------------------------------------------------------- Q2
def to_8bit(path, lo=2.0, hi=99.5, bands=(1, 2, 3)):
    """12-bit scene -> the 0-255 range the model was trained on.  The window is
    returned too, because the result cannot be undone without it."""
    with rasterio.open(path) as d:
        a = d.read(bands).astype(np.float32)
    w_lo = np.percentile(a, lo, axis=(1, 2))[:, None, None]
    w_hi = np.percentile(a, hi, axis=(1, 2))[:, None, None]
    out = (np.clip((a - w_lo) / (w_hi - w_lo), 0, 1) * 255).round().astype(np.uint8)
    return np.transpose(out, (1, 2, 0)), (w_lo.ravel(), w_hi.ravel())


def show_conversion(path, net, cfg, lo=2.0, hi=99.5):
    """The same scene through the model twice: converted, and not."""
    lr, _ = to_8bit(path, lo, hi)
    with rasterio.open(path) as d:
        raw = np.transpose(d.read((1, 2, 3)), (1, 2, 0)).astype(np.float32)

    good = sr_tiled(lr, net, cfg)
    x = torch.from_numpy(raw[None]).permute(0, 3, 1, 2).float().to(cfg.device)
    with torch.no_grad():
        y = net(x)[0].cpu().numpy()
    over = 100 * (y > 255).mean()
    bad = np.clip(y, 0, 255).astype(np.uint8).transpose(1, 2, 0)

    fig, axes = plt.subplots(1, 3, figsize=(16.2, 6.0), layout='constrained')
    for ax, im, t, lab, col in (
            (axes[0], lr, '1.  input, converted', f'0 - 255', C_OK),
            (axes[1], good, '2.  SR from that', f'{good.min()} - {good.max()}', C_OK),
            (axes[2], bad, '3.  SR from the raw 12-bit',
             f'{over:.0f}% of the output went over 255', C_BAD)):
        ax.imshow(im)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_box_aspect(1)
        ax.set_title(t, fontsize=21, pad=12, loc='left')
        ax.set_xlabel(lab, fontsize=18, color=col)
        for sp in ax.spines.values():
            sp.set(color=col, linewidth=3.0)
    plt.show()


# --------------------------------------------------------------- Q5
def save_sr(src_path, arr, out_path, scale):
    """Write the result with the input's CRS and origin, pixels `scale` times
    finer, so the two overlay exactly."""
    with rasterio.open(src_path) as d:
        prof = d.profile
        t = d.transform
    prof.update(height=arr.shape[0], width=arr.shape[1], count=arr.shape[2],
                dtype=arr.dtype, compress='deflate',
                transform=rasterio.Affine(t.a / scale, t.b, t.c,
                                          t.d, t.e / scale, t.f))
    with rasterio.open(out_path, 'w', **prof) as d:
        d.write(np.transpose(arr, (2, 0, 1)))
    return out_path
