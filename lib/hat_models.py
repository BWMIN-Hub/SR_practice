"""HAT 불러오기 도우미. 신경망 정의는 업스트림 사본(hat_arch.py) 그대로다.

    from hat_models import load_hat
    net = load_hat('hat_x3.pth')

HAT 은 ×3 을 기본 지원한다. SwinIR 과 같고, SRGAN·ESRGAN 처럼 업샘플을 고칠 필요가 없다.
einops 와 timm 이 필요하다.
"""
import numpy as np
import torch

from hat_arch import HAT

# 논문 HAT 기본 설정 (window 16, HAB + OCAB)
CONFIG = dict(in_chans=3, img_size=64, window_size=16, compress_ratio=3,
              squeeze_factor=30, conv_scale=0.01, overlap_ratio=0.5, img_range=1.,
              depths=[6] * 6, embed_dim=180, num_heads=[6] * 6, mlp_ratio=2,
              upsampler='pixelshuffle', resi_connection='1conv')


def build_hat(scale=3, **kw):
    cfg = dict(CONFIG, upscale=scale)
    cfg.update(kw)
    return HAT(**cfg)


def load_hat(weight, scale=3, device=None, **kw):
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    net = build_hat(scale, **kw).to(device)
    net.load_state_dict(torch.load(weight, map_location=device))
    return net.eval()


@torch.no_grad()
def _forward(net, lr):
    """한 조각 추론. HAT 은 입력을 window 배수로 맞춰주지 않아 반사 패딩이 필요하다."""
    import torch.nn.functional as F
    dev = next(net.parameters()).device
    x = torch.from_numpy(lr.transpose(2, 0, 1)).float()[None].to(dev) / 255
    h, w = x.shape[-2:]
    win = net.window_size
    ph, pw = (win - h % win) % win, (win - w % win) % win
    if ph or pw:
        x = F.pad(x, (0, pw, 0, ph), mode='reflect')
    s = net.upscale
    return net(x)[:, :, :h * s, :w * s].clamp(0, 1)[0].cpu().numpy().transpose(1, 2, 0)


@torch.no_grad()
def hat_upscale(net, lr, tile=192, overlap=32):
    """HAT 추론. (H,W,3) uint8 -> (H*s,W*s,3) uint8.

    HAT 은 어텐션이라 입력이 커지면 메모리가 급격히 는다. 인천(600px)을 통째로
    넣으면 13.8 GB 를 써서 Colab T4(15 GB)에서 터진다. 그래서 기본으로
    **타일을 나눠 추론하고 겹치는 부분을 평균**낸다. tile=None 이면 통째로 한다.

    tile 은 window(16) 의 배수로 두는 편이 좋다. 안 맞으면 조각마다 반사 패딩이 붙는다.

    타일 결과는 통짜와 완전히 같지는 않다(평균 1.4/255, 0.55%). 이음매 탓이 아니라
    어텐션이 보는 문맥이 좁아져서 생기는 차이라, 겹침을 늘려도 거의 줄지 않는다.
    검증 패치(128px)는 타일보다 작아 통짜 경로를 그대로 타므로 점수는 영향이 없다.
    """
    h, w = lr.shape[:2]
    s = net.upscale
    if tile is None or (h <= tile and w <= tile):
        out = _forward(net, lr)
        return (out * 255).round().astype('uint8')

    acc = np.zeros((h * s, w * s, 3), np.float32)
    cnt = np.zeros((h * s, w * s, 1), np.float32)
    step = max(1, tile - overlap)
    ys = list(range(0, max(1, h - tile + 1), step))
    xs = list(range(0, max(1, w - tile + 1), step))
    if ys[-1] != h - tile and h > tile:
        ys.append(h - tile)
    if xs[-1] != w - tile and w > tile:
        xs.append(w - tile)

    for y in ys:
        for x in xs:
            ph, pw = min(tile, h - y), min(tile, w - x)
            sr = _forward(net, lr[y:y + ph, x:x + pw])
            acc[y * s:(y + ph) * s, x * s:(x + pw) * s] += sr
            cnt[y * s:(y + ph) * s, x * s:(x + pw) * s] += 1

    return (np.clip(acc / np.maximum(cnt, 1), 0, 1) * 255).round().astype('uint8')
