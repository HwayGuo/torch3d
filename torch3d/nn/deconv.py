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


class PointDeconv(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size=1, bandwidth=1.0, bias=True
    ):
        super(PointDeconv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = _single(out_channels)
        self.kernel_size = kernel_size
        self.bandwidth = bandwidth
        self.bias = bias
        self.scale = nn.Sequential(
            nn.Conv1d(1, 8, 1, bias=self.bias),
            nn.BatchNorm1d(8),
            nn.ReLU(True),
            nn.Conv1d(8, 8, 1, bias=self.bias),
            nn.BatchNorm1d(8),
            nn.ReLU(True),
            nn.Conv1d(8, 1, 1, bias=self.bias),
            nn.Sigmoid(),
        )
        self.weight = nn.Sequential(
            nn.Conv2d(3, 8, 1, bias=self.bias),
            nn.BatchNorm2d(8),
            nn.ReLU(True),
            nn.Conv2d(8, 8, 1, bias=self.bias),
            nn.BatchNorm2d(8),
            nn.ReLU(True),
            nn.Conv2d(8, 16, 1, bias=self.bias),
        )
        in_channels = self.in_channels
        modules = []
        for channels in self.out_channels[:-1]:
            modules.append(nn.Conv2d(in_channels, channels, 1, bias=self.bias))
            modules.append(nn.BatchNorm2d(channels))
            modules.append(nn.ReLU(True))
            in_channels = channels
        self.mlp = nn.Sequential(*modules)
        self.lin = nn.Sequential(
            nn.Conv2d(in_channels, self.out_channels[-1], [16, 1], bias=self.bias),
            nn.BatchNorm2d(self.out_channels[-1]),
            nn.ReLU(True),
        )

    def forward(self, x, y):
        batch_size = x.shape[0]
        p, x = x[:, :3], x[:, 3:]
        q, y = y[:, :3], y[:, 3:]
        x = F.interpolate(p, q, x, self.kernel_size)
        x = torch.cat([x, y], dim=1)
        s = F.kernel_density(q, self.bandwidth).unsqueeze(1)
        s = self.scale(torch.reciprocal(s))  # calculate scaling factor
        _, index = F.knn(q, q, self.kernel_size)
        index = index.view(batch_size, -1).unsqueeze(1)
        # Point and density grouping
        p = torch.gather(q, 2, index.expand(-1, 3, -1))
        x = torch.gather(x, 2, index.expand(-1, self.in_channels, -1))
        s = torch.gather(s, 2, index)
        p = p.view(batch_size, 3, self.kernel_size, -1)
        x = x.view(batch_size, self.in_channels, self.kernel_size, -1)
        s = s.view(batch_size, 1, self.kernel_size, -1)
        p = p - q.unsqueeze(2).expand(-1, -1, self.kernel_size, -1)
        w = self.weight(p)
        x = self.mlp(x * s)
        x = torch.matmul(w.permute(0, 3, 1, 2), x.permute(0, 3, 2, 1))
        x = x.permute(0, 3, 2, 1)
        x = self.lin(x).squeeze(2)
        x = torch.cat([q, x], dim=1)
        return x
