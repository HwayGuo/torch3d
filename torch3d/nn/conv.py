import torch
import torch.nn as nn
from collections.abc import Iterable
from torch3d.nn import functional as F
from math import ceil


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


class XConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, bias=True):
        super(XConv, self).__init__()
        self.in_channels = in_channels - 3
        self.out_channels = out_channels
        self.mid_channels = out_channels // 4
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.bias = bias
        self.mlp = nn.Sequential(
            nn.Conv2d(3, self.mid_channels, 1, bias=self.bias),
            nn.BatchNorm2d(self.mid_channels),
            nn.ReLU(True),
            nn.Conv2d(self.mid_channels, self.mid_channels, 1, bias=self.bias),
            nn.BatchNorm2d(self.mid_channels),
            nn.ReLU(True),
        )
        self.stn = nn.Sequential(
            nn.Conv2d(3, self.kernel_size ** 2, [1, self.kernel_size], bias=self.bias),
            nn.BatchNorm2d(self.kernel_size ** 2),
            nn.ReLU(True),
            # nn.Conv2d(self.kernel_size ** 2, self.kernel_size ** 2, 1, bias=self.bias),
            # nn.BatchNorm2d(self.kernel_size ** 2),
            # nn.ReLU(True),
            # nn.Conv2d(self.kernel_size ** 2, self.kernel_size ** 2, 1, bias=self.bias),
            # nn.BatchNorm2d(self.kernel_size ** 2),
        )
        in_channels = self.in_channels + self.mid_channels
        dm = int(ceil(out_channels / in_channels))
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels,
                in_channels * dm,
                [1, self.kernel_size],
                groups=in_channels,
            ),
            nn.Conv2d(in_channels * dm, self.out_channels, 1, bias=self.bias),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU(True),
        )

    def forward(self, p, q, x=None):
        batch_size = p.shape[0]
        _, index = F.knn(p, q, self.kernel_size * self.dilation)
        index = index[..., :: self.dilation]
        index = index.view(batch_size, -1).unsqueeze(1)
        p_hat = torch.gather(p, 2, index.expand(-1, 3, -1))
        p_hat = p_hat.view(batch_size, 3, -1, self.kernel_size)
        p_hat = p_hat - q.unsqueeze(3).expand(-1, -1, -1, self.kernel_size)
        x_hat = self.mlp(p_hat)
        if x is not None:
            x = torch.gather(x, 2, index.expand(-1, self.in_channels, -1))
            x = x.view(batch_size, self.in_channels, -1, self.kernel_size)
            x_hat = torch.cat([x_hat, x], dim=1)
        print(p_hat.shape)
        T = self.stn(p_hat)
        print(T.shape)
        T = T.view(batch_size, self.kernel_size, self.kernel_size, -1)
        T = T.permute(0, 3, 1, 2).unsqueeze(1)
        x = torch.matmul(T, x_hat.unsqueeze(4)).squeeze(4)
        x = self.conv(x).squeeze(3)
        return q, x


if __name__ == "__main__":
    p = torch.rand(1, 3, 1024)
    q = torch.rand(1, 3, 512)
    m = XConv(3, 48, 8, dilation=1, bias=False)
    m(p, q)
