"""SRGAN GeneratorLoss (업스트림과 동일한 계산식).

    total = image + 0.001*adversarial + 0.006*perception + 2e-8*tv

항별 값은 self.last 에 남는다 — 어느 항이 학습을 끄는지 보려면 이걸 본다.
"""
import torch
import torch.nn as nn
from torchvision.models import vgg16

W = {'image': 1.0, 'adversarial': 0.001, 'perception': 0.006, 'tv': 2e-8}


class TVLoss(nn.Module):
    def forward(self, x):
        h = torch.pow(x[:, :, 1:, :] - x[:, :, :-1, :], 2).mean()
        w = torch.pow(x[:, :, :, 1:] - x[:, :, :, :-1], 2).mean()
        return 2 * (h + w)


class GeneratorLoss(nn.Module):
    def __init__(self):
        super().__init__()
        net = nn.Sequential(*list(vgg16(weights='DEFAULT').features)[:31]).eval()
        for p in net.parameters():
            p.requires_grad = False
        self.loss_network, self.mse_loss, self.tv_loss = net, nn.MSELoss(), TVLoss()

    def forward(self, out_labels, out_images, target_images):
        image = self.mse_loss(out_images, target_images)
        adversarial = torch.mean(1 - out_labels)
        perception = self.mse_loss(self.loss_network(out_images),
                                   self.loss_network(target_images))
        tv = self.tv_loss(out_images)
        self.last = {'image': image.item(), 'adversarial': adversarial.item(),
                     'perception': perception.item(), 'tv': tv.item()}
        return (W['image'] * image + W['adversarial'] * adversarial
                + W['perception'] * perception + W['tv'] * tv)
