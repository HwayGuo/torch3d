import torch
import torch.nn as nn
from torch3d.nn import functional as F
from torch3d.nn.utils import _single


class SetDeconv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, bias=True):
        self.in_channels = in_channels
        self.out_channels = _single(out_channels)
        self.kernel_size = kernel_size
        self.bias = bias
        in_channels = self.in_channels
        modules = []
        for channels in self.out_channels:
            modules.append(nn.Conv1d(in_channels, channels, 1, bias=self.bias))
            modules.append(nn.BatchNorm1d(channels))
            modules.append(nn.ReLU(True))
            in_channels = channels
        super(SetDeconv, self).__init__(*modules)

    def forward(self, x, y):
        p, x = x[:, :3], x[:, 3:]
        q, y = y[:, :3], y[:, 3:]
        x = F.interpolate(p, q, x, self.kernel_size)
        x = torch.cat([x, y], dim=1)
        x = super(SetDeconv, self).forward(x)
        x = torch.cat([q, x], dim=1)
        return x
