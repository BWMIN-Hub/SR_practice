"""SwinIR 불러오기 도우미. 신경망 정의는 업스트림 파일(swinir_arch.py) 그대로다.

    from swinir_models import load_swinir
    net = load_swinir('swinir_x3.pth')

SwinIR 은 ×3 을 기본 지원한다 (Upsample 클래스가 scale==3 을 따로 처리).
SRGAN·ESRGAN 처럼 업샘플 부분을 고칠 필요가 없었다.
"""
import torch

from swinir_arch import SwinIR

# 논문 classical SR 설정
CLASSICAL = dict(in_chans=3, img_size=48, window_size=8, img_range=1.,
                 depths=[6] * 6, embed_dim=180, num_heads=[6] * 6,
                 mlp_ratio=2, upsampler='pixelshuffle', resi_connection='1conv')


def build_swinir(scale=3, **kw):
    cfg = dict(CLASSICAL, upscale=scale)
    cfg.update(kw)
    return SwinIR(**cfg)


def load_swinir(weight, scale=3, device=None, **kw):
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    net = build_swinir(scale, **kw).to(device)
    net.load_state_dict(torch.load(weight, map_location=device))
    return net.eval()
