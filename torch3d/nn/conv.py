import torch
import torch.nn as nn
from torch3d.nn import functional as F
from torch3d.nn.utils import _single


class PointConv(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=1,
        stride=1,
        bandwidth=1,
        radius=None,
        bias=True,
    ):
        self.in_channels = in_channels
        self.out_channels = _single(out_channels)
        self.kernel_size = kernel_size
        self.bias = bias


class EdgeConv(nn.Sequential):
    """
    The edge convolution layer introduced in the `"Dynamic Graph CNN for Learning on Point Clouds"
    <https://arxiv.org/abs/1801.07829>`_ paper

    Args:
        in_channels (int): Number of channels in the input point set
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int): Neighborhood size of the convolution kernel
        bias (bool, optional): If ``True``, adds a learnable bias to the output. Default: ``True``
    """

    def __init__(self, in_channels, out_channels, kernel_size, bias=True):
        self.in_channels = in_channels
        self.out_channels = _single(out_channels)
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


class SetConv(nn.Sequential):
    """
    The set abstraction layer introduced in the `"PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space"
    <https://arxiv.org/abs/1706.02413>`_ paper

    Args:
        in_channels (int): Number of channels in the input point set
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int, optional): Neighborhood size of the convolution kernel. Default: 1
        stride (int, optional): Reduction rate of farthest point sampling. Default: 1
        radius (float, optional): Radius for the neighborhood search. Default: ``None``
        bias (bool, optional): If ``True``, adds a learnable bias to the output. Default: ``True``
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=1,
        stride=1,
        radius=None,
        bias=True,
    ):
        self.in_channels = in_channels
        self.out_channels = _single(out_channels)
        self.kernel_size = kernel_size
        self.stride = stride
        self.radius = radius
        self.bias = bias
        in_channels = self.in_channels
        modules = []
        for channels in self.out_channels:
            modules.append(nn.Conv2d(in_channels, channels, 1, bias=self.bias))
            modules.append(nn.BatchNorm2d(channels))
            modules.append(nn.ReLU(True))
            in_channels = channels
        modules.append(nn.AdaptiveMaxPool2d([1, None]))
        super(SetConv, self).__init__(*modules)

    def forward(self, x):
        batch_size = x.shape[0]
        num_points = x.shape[2]
        if self.radius is not None:
            num_samples = num_points // self.stride
            p = x[:, :3]  # XYZ coordinates
            q = F.farthest_point_sample(p, num_samples)
            index = F.ball_point(p, q, self.kernel_size, self.radius)
            index = index.view(batch_size, -1)
            index = index.unsqueeze(1).expand(-1, self.in_channels, -1)
            x = torch.gather(x, 2, index)
            x = x.view(batch_size, self.in_channels, self.kernel_size, -1)
            x[:, :3] -= q.unsqueeze(2).expand(-1, -1, self.kernel_size, -1)
            x = super(SetConv, self).forward(x)
            x = x.squeeze(2)
            x = torch.cat([q, x], dim=1)
        else:
            x = x.unsqueeze(3)
            x = super(SetConv, self).forward(x)
            x = x.squeeze(2)
        return x
