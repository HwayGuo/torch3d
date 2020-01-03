import torch
import torch.nn as nn
from torch3d.nn import functional as F


class ChamferLoss(nn.Module):
    def __init__(self):
        super(ChamferLoss, self).__init__()

    def forward(self, x, y):
        return F.chamfer_loss(x, y)
