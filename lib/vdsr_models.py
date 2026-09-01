"""VDSR 불러오기 도우미. 신경망 정의는 업스트림 사본(vdsr_arch.py) 그대로다.

    from vdsr_models import load_vdsr, vdsr_upscale
    net = load_vdsr('vdsr_x3.pth')
    sr  = vdsr_upscale(net, lr)

SRCNN 과 규약이 같다. 모델 안에 업샘플이 없어 LR 을 미리 bicubic 으로 3배 키워 넣고,
Y(밝기) 채널 하나만 본다. 색은 bicubic 확대본을 그대로 쓴다.

다른 점은 **잔차 학습**이다. conv 20층이 입력과의 차이만 배우고 마지막에 입력을 더한다
(`out = torch.add(out, identity)`). 그래서 학습 시작부터 출력이 bicubic 수준이고,
높은 학습률(0.1) + 기울기 자르기를 쓸 수 있다.
"""
import numpy as np
import torch

import vdsr_imgproc as imgproc
from vdsr_arch import VDSR


def build_vdsr(**kw):
    return VDSR(**kw)


def load_vdsr(weight, device=None, **kw):
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    net = build_vdsr(**kw).to(device)
    sd = torch.load(weight, map_location=device)
    net.load_state_dict(sd.get('state_dict', sd))
    return net.eval()


@torch.no_grad()
def vdsr_upscale(net, lr, scale=3):
    """VDSR 추론. (H,W,3) uint8 -> (H*s,W*s,3) uint8."""
    dev = next(net.parameters()).device
    up = imgproc.imresize(np.asarray(lr, np.float32) / 255., scale)     # RGB float [0,1]
    ycbcr = imgproc.rgb2ycbcr(up, use_y_channel=False)
    y = torch.from_numpy(np.ascontiguousarray(ycbcr[..., 0])).float()[None, None].to(dev)

    ycbcr[..., 0] = net(y).clamp(0, 1)[0, 0].cpu().numpy()
    return (np.clip(imgproc.ycbcr2rgb(ycbcr), 0, 1) * 255).round().astype('uint8')


def vdsr_pairs(lr_list, hr_list, scale=3):
    """학습용 텐서. RGB uint8 목록 -> (입력 Y, 목표 Y). LR 은 실제 관측 영상 그대로다."""
    def y(img, up):
        a = np.asarray(img, np.float32) / 255.
        if up:
            a = imgproc.imresize(a, scale)
        return imgproc.rgb2ycbcr(a, use_y_channel=True)

    x = np.stack([y(a, True) for a in lr_list])[:, None]
    t = np.stack([y(a, False) for a in hr_list])[:, None]
    return torch.from_numpy(x).float(), torch.from_numpy(t).float()
