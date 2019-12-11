import torch
import torch.nn as nn
from torch3d.nn import functional as F


__all__ = ["ChamferLoss", "DiscriminativeLoss"]


class ChamferLoss(nn.Module):
    def __init__(self):
        super(ChamferLoss, self).__init__()

    def forward(self, x, y):
        return F.chamfer_loss(x, y)


class DiscriminativeLoss(nn.Module):
    def __init__(
        self, alpha=1.0, beta=1.0, gamma=1.0, delta_v=0.5, delta_d=1.5, reduction="mean"
    ):
        super(DiscriminativeLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta_d = delta_d
        self.delta_v = delta_v
        self.reduction = reduction

    def forward(self, x, y):
        return F.discriminative_loss(
            x,
            y,
            self.alpha,
            self.beta,
            self.gamma,
            self.delta_v,
            self.delta_d,
            self.reduction,
        )
