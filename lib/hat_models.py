"""HAT 불러오기 도우미. 신경망 정의는 업스트림 사본(hat_arch.py) 그대로다.

    from hat_models import load_hat
    net = load_hat('hat_x3.pth')

HAT 은 ×3 을 기본 지원한다. SwinIR 과 같고, SRGAN·ESRGAN 처럼 업샘플을 고칠 필요가 없다.
einops 와 timm 이 필요하다.
"""
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
def hat_upscale(net, lr):
    """HAT 추론. (H,W,3) uint8 -> (H*s,W*s,3) uint8.

    HAT 은 SwinIR 과 달리 입력을 window 배수로 맞춰주지 않는다.
    (예: 600px 은 16 으로 안 나뉘어 그대로 넣으면 shape 오류가 난다)
    반사 패딩으로 맞춘 뒤 결과를 원래 크기로 잘라낸다.
    """
    import torch.nn.functional as F
    dev = next(net.parameters()).device
    x = torch.from_numpy(lr.transpose(2, 0, 1)).float()[None].to(dev) / 255
    h, w = x.shape[-2:]
    win = net.window_size
    ph, pw = (win - h % win) % win, (win - w % win) % win
    if ph or pw:
        x = F.pad(x, (0, pw, 0, ph), mode='reflect')
    s = net.upscale
    out = net(x)[:, :, :h * s, :w * s].clamp(0, 1)
    return (out[0].cpu().numpy().transpose(1, 2, 0) * 255).round().astype('uint8')
