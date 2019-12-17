import torch
import torch.nn as nn
from collections.abc import Iterable
from torch3d.nn import functional as F


class EdgeConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, bias=True):
        self.in_channels = in_channels
        if not isinstance(out_channels, Iterable):
            self.out_channels = (out_channels,)
        else:
            self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.bias = bias
        in_channels = in_channels * 2
        modules = []
        for channels in self.out_channels:
            modules.append(nn.Conv2d(in_channels, channels, 1, bias=self.bias))
            modules.append(nn.BatchNorm2d(channels))
            modules.append(nn.ReLU(True))
            in_channels = channels
        modules.append(nn.MaxPool2d([self.kernel_size, 1]))
        super(EdgeConv, self).__init__(*modules)

    def forward(self, x):
        batch_size = x.shape[0]
        _, index = F.knn(x, x, self.kernel_size)
        index = index.view(batch_size, -1).unsqueeze(1)
        index = index.expand(-1, self.in_channels, -1)
        x_hat = torch.gather(x, 2, index)
        x_hat = x_hat.view(batch_size, self.in_channels, self.kernel_size, -1)
        x = x.unsqueeze(2).expand(-1, -1, self.kernel_size, -1)
        x_hat = x_hat - x
        x = torch.cat([x, x_hat], dim=1)
        x = super(EdgeConv, self).forward(x)
        x = x.squeeze(2)
        return x


class SetConv(nn.Module):
    def __init__(self, in_channels, mlp, radius=None, k=None, bias=True):
        super(SetConv, self).__init__()
        self.in_channels = in_channels
        self.radius = radius
        self.k = k
        self.bias = bias
        modules = []
        last_channels = self.in_channels
        for channels in mlp:
            modules.append(nn.Conv2d(last_channels, channels, 1, bias=self.bias))
            modules.append(nn.BatchNorm2d(channels))
            modules.append(nn.ReLU(True))
            last_channels = channels
        self.mlp = nn.Sequential(*modules)
        self.maxpool = nn.MaxPool2d([1, k])

    def forward(self, p, q, x=None):
        batch_size = p.shape[0]
        if self.radius is not None:
            index = F.ball_point(p, q, self.k, self.radius)
            index = index.view(batch_size, -1)
            p = torch.gather(p, 1, index.unsqueeze(2).expand(-1, -1, 3))
            p = p.view(batch_size, -1, self.k, 3)
            p_hat = p - q.unsqueeze(2)
            x_hat = p_hat
        else:
            x_hat = p.unsqueeze(1)
        if x is not None:
            x = x.permute(0, 2, 1)
            if self.radius:
                x = torch.gather(x, 1, index.unsqueeze(2).expand(-1, -1, x.shape[2]))
                x = x.view(batch_size, -1, self.k, x.shape[2])
            else:
                x = x.unsqueeze(1)
            x_hat = torch.cat([x_hat, x], dim=-1)
        x = x_hat.permute(0, 3, 1, 2)
        x = self.mlp(x)
        x = self.maxpool(x).squeeze(3)
        return q, x
