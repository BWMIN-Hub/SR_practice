"""SRCNN 불러오기 도우미. 신경망 정의는 업스트림 사본(srcnn_arch.py) 그대로다.

    from srcnn_models import load_srcnn, srcnn_upscale
    net = load_srcnn('srcnn_x3.pth')
    sr  = srcnn_upscale(net, lr)

SRCNN 은 앞선 모델들과 규약이 두 가지 다르다.

1. **모델 안에 업샘플이 없다.** conv 3장이 전부다. 입력을 미리 bicubic 으로 3배
   키워서 넣는다. 배율은 모델이 아니라 이 전처리가 정한다.
2. **Y(밝기) 채널 하나만 본다.** 색(Cb, Cr)은 bicubic 확대본을 그대로 쓴다.
   그래서 SRCNN 은 색을 바꾸지 못하고 선명도만 손본다.

bicubic 과 YCbCr 변환은 업스트림 imgproc 을 쓴다 — 학습 때와 같아야 한다.
"""
import numpy as np
import torch

import srcnn_imgproc as imgproc
from srcnn_arch import SRCNN


def build_srcnn(**kw):
    return SRCNN(**kw)


def load_srcnn(weight, device=None, **kw):
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    net = build_srcnn(**kw).to(device)
    sd = torch.load(weight, map_location=device)
    net.load_state_dict(sd.get('state_dict', sd))
    return net.eval()


@torch.no_grad()
def srcnn_upscale(net, lr, scale=3):
    """SRCNN 추론. (H,W,3) uint8 -> (H*s,W*s,3) uint8.

    LR 을 bicubic 으로 키운 뒤 Y 채널만 신경망에 통과시키고, 색은 확대본에서 가져와
    다시 합친다. 업스트림 inference.py 와 같은 순서다.
    """
    dev = next(net.parameters()).device
    up = imgproc.image_resize(np.asarray(lr, np.float32) / 255., scale)   # RGB float [0,1]
    ycbcr = imgproc.rgb2ycbcr(up, only_use_y_channel=False)
    y = torch.from_numpy(np.ascontiguousarray(ycbcr[..., 0])).float()[None, None].to(dev)

    sr_y = net(y).clamp(0, 1)[0, 0].cpu().numpy()
    ycbcr[..., 0] = sr_y
    rgb = imgproc.ycbcr2rgb(ycbcr)
    return (np.clip(rgb, 0, 1) * 255).round().astype('uint8')


@torch.no_grad()
def srcnn_bicubic(lr, scale=3):
    """비교용 — SRCNN 이 입력으로 받는 그 bicubic 확대본 (uint8)."""
    up = imgproc.image_resize(np.asarray(lr, np.float32) / 255., scale)
    return (np.clip(up, 0, 1) * 255).round().astype('uint8')


def srcnn_pairs(lr_list, hr_list, scale=3):
    """학습용 텐서 만들기. RGB uint8 목록 -> (입력 Y, 목표 Y).

    입력은 bicubic 으로 키운 LR 의 Y 채널, 목표는 HR 의 Y 채널이다.
    HR 을 줄여서 LR 을 만들지 않는다 — LR 은 실제 관측 영상 그대로다.
    """
    def y(img, up):
        a = np.asarray(img, np.float32) / 255.
        if up:
            a = imgproc.image_resize(a, scale)
        return imgproc.rgb2ycbcr(a, only_use_y_channel=True)

    x = np.stack([y(a, True) for a in lr_list])[:, None]
    t = np.stack([y(a, False) for a in hr_list])[:, None]
    return torch.from_numpy(x).float(), torch.from_numpy(t).float()
