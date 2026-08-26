"""SR 학습용 손실 함수. `from sr_losses import get_loss` 로 쓴다."""
import torch
import torch.nn as nn


class CharbonnierLoss(nn.Module):
    """L1 의 부드러운 판. 0 근처에서 미분이 안정적이다."""

    def __init__(self, eps=1e-3):
        super().__init__()
        self.eps2 = eps ** 2

    def forward(self, pred, target):
        return torch.sqrt((pred - target) ** 2 + self.eps2).mean()


LOSSES = {
    'l1': nn.L1Loss,
    'l2': nn.MSELoss,
    'charbonnier': CharbonnierLoss,
}


def get_loss(name='l1'):
    key = name.lower()
    if key not in LOSSES:
        raise ValueError(f'모르는 손실: {name}. 가능한 값: {list(LOSSES)}')
    return LOSSES[key]()
