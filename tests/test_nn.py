import torch
import torch3d.nn as nn
import torch3d.nn.functional as F


def test_conv():
    names = ["EdgeConv", "SetConv", "PointConv"]
    batch_size = 2
    in_channels = 3
    out_channels = 64
    kernel_size = 32
    num_points = 1024
    stride = 1
    radius = 0.1
    bandwidth = 0.1
    x = torch.rand(batch_size, in_channels, num_points)

    for name in names:
        cls = getattr(nn, name)
        if name == "SetConv":
            size = torch.Size([batch_size, out_channels + 3, num_points])
            conv = cls(in_channels, out_channels, kernel_size, stride, radius)
        elif name == "PointConv":
            size = torch.Size([batch_size, out_channels + 3, num_points])
            conv = cls(in_channels, out_channels, kernel_size, stride, bandwidth)
        elif name == "EdgeConv":
            size = torch.Size([batch_size, out_channels, num_points])
            conv = cls(in_channels, out_channels, kernel_size)
        assert conv(x).shape == size


def test_deconv():
    names = ["SetDeconv"]
    batch_size = 2
    in_channels = 32
    out_channels = 64
    kernel_size = 3
    num_points = 1024
    x = torch.rand(batch_size, in_channels + 3, num_points)
    y = x.clone()
    size = torch.Size([batch_size, out_channels + 3, num_points])

    for name in names:
        cls = getattr(nn, name)
        dconv = cls(in_channels * 2, out_channels, kernel_size)
        assert dconv(x, y).shape == size


def test_farthest_point_sample():
    batch_size = 1
    num_samples = 2
    in_channels = 3
    p = torch.tensor(
        [
            [
                [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
                [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
                [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0],
            ]
        ]
    )
    size = torch.Size([batch_size, in_channels, num_samples])
    q = F.farthest_point_sample(p, num_samples)
    assert q.shape == size
    assert q.tolist() == [[[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]]]


def test_ball_point():
    batch_size = 1
    k = 2
    radius = 0.5
    p = torch.tensor(
        [
            [
                [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
                [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
                [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0],
            ]
        ]
    )
    q = p.clone()
    num_points = p.shape[2]
    size = torch.Size([batch_size, k, num_points])
    index = F.ball_point(p, q, k, radius)
    assert index.shape == size
    assert index.tolist() == [[[0, 1, 2, 3, 4, 5, 6, 7], [0, 1, 2, 3, 4, 5, 6, 7]]]


def test_interpolate():
    k = 2
    p = torch.tensor(
        [
            [
                [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
                [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
                [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0],
            ]
        ]
    )
    q = p.clone()
    x = torch.tensor([[[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]]])
    y = F.interpolate(p, q, x, k)
    assert torch.allclose(x, y)
