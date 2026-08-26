"""SR 모델 정의. 실습 노트북에서 `from sr_models import load_edsr` 로 쓴다."""
import torch
import torch.nn as nn


def conv(i, o, k=3):
    return nn.Conv2d(i, o, k, padding=k // 2)


class MeanShift(nn.Conv2d):
    """입력에서 평균색을 빼고 출력에서 도로 더한다."""

    def __init__(self, rgb_range=255, sign=-1):
        super().__init__(3, 3, 1)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1)
        self.bias.data = sign * rgb_range * torch.tensor([0.4488, 0.4371, 0.4040])
        for p in self.parameters():
            p.requires_grad = False


class ResBlock(nn.Module):
    def __init__(self, n_feats):
        super().__init__()
        self.body = nn.Sequential(conv(n_feats, n_feats), nn.ReLU(True),
                                  conv(n_feats, n_feats))

    def forward(self, x):
        return self.body(x) + x


class EDSR(nn.Module):
    """EDSR baseline. 계층 이름이 배포된 체크포인트 키와 일치해야 한다."""

    def __init__(self, n_resblocks=16, n_feats=64, scale=3):
        super().__init__()
        self.sub_mean = MeanShift(255, -1)
        self.add_mean = MeanShift(255, +1)
        self.head = nn.Sequential(conv(3, n_feats))
        self.body = nn.Sequential(*[ResBlock(n_feats) for _ in range(n_resblocks)],
                                  conv(n_feats, n_feats))
        self.tail = nn.Sequential(
            nn.Sequential(conv(n_feats, n_feats * scale ** 2), nn.PixelShuffle(scale)),
            conv(n_feats, 3))

    def forward(self, x):
        x = self.head(self.sub_mean(x))
        return self.add_mean(self.tail(self.body(x) + x))


def load_edsr(weight=None, device=None, **kw):
    """모델을 만들고 (주어지면) 가중치를 실어 평가 모드로 돌려준다."""
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    net = EDSR(**kw).to(device)
    if weight:
        net.load_state_dict(torch.load(weight, map_location=device))
    return net.eval()
