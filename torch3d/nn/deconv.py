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
        batch_size = x.shape[0]
        channels = x.shape[1]
        sqdist, index = F.knn(p, q, self.kernel_size)
        sqdist = torch.clamp(sqdist, min=1e-10)
        weight = torch.reciprocal(sqdist)
        weight = weight / torch.sum(weight, dim=1, keepdim=True)
        weight = weight.unsqueeze(1)
        index = index.view(batch_size, -1)
        index = index.unsqueeze(1).expand(-1, channels, -1)
        x = torch.gather(x, 2, index)
        x = x.view(batch_size, channels, self.kernel_size, -1)
        x = torch.sum(x * weight, dim=2)
        x = torch.cat([x, y], dim=1)
        x = super(SetDeconv, self).forward(x)
        return x
