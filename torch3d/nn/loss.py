import torch
import torch.nn as nn
from torch3d.nn import functional as F


__all__ = ["ChamferLoss", "DiscriminativeLoss"]


class ChamferLoss(nn.Module):
    def __init__(self):
        super(ChamferLoss, self).__init__()

    def forward(self, input, target):
        return F.chamfer_loss(input, target)


class DiscriminativeLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=1.0, gamma=1.0, delta_v=0.5, delta_d=1.5):
        super(DiscriminativeLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta_d = delta_d
        self.delta_v = delta_v

    def forward(self, input, target):
        return F.discriminative_loss(
            input, target, self.alpha, self.beta, self.gamma, self.delta_v, self.delta_d
        )
