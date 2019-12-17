import torch
import torch3d.nn as nn


def test_conv():
    names = [
        "EdgeConv",
    ]
    batch_size = 2
    in_channels = 3
    out_channels = 64
    kernel_size = 32
    num_points = 1024
    x = torch.rand(batch_size, in_channels, num_points)
    size = torch.Size([batch_size, out_channels, num_points])

    for name in names:
        cls = getattr(nn, name)
        conv = cls(in_channels, out_channels, kernel_size)
        assert conv(x).shape == size
