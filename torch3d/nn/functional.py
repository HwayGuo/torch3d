import math
import torch
import torch.nn.functional as F
from torch3d.extension import _lazy_import


def cdist(x, y):
    xx = x.pow(2).sum(dim=1, keepdim=True).permute(0, 2, 1)
    yy = y.pow(2).sum(dim=1, keepdim=True)
    sqdist = torch.baddbmm(yy, x.permute(0, 2, 1), y, alpha=-2).add_(xx)
    return sqdist


def knn(p, q, k):
    sqdist = cdist(p, q)
    return torch.topk(sqdist, k, dim=1, largest=False)


def ball_point(p, q, k, radius):
    _C = _lazy_import()
    return _C.ball_point(p.contiguous(), q.contiguous(), k, radius)


def farthest_point_sample(p, num_samples):
    in_channels = p.shape[1]
    num_points = p.shape[2]
    if num_samples > num_points:
        raise ValueError("num_samples should be less than input size.")
    _C = _lazy_import()
    index = _C.farthest_point_sample(p.contiguous(), num_samples)
    index = index.unsqueeze(1).expand(-1, in_channels, -1)
    return torch.gather(p, 2, index)


def interpolate(p, q, x, k):
    batch_size = x.shape[0]
    in_channels = x.shape[1]
    sqdist, index = knn(p, q, k)
    sqdist = torch.clamp(sqdist, min=1e-10)
    weight = torch.reciprocal(sqdist)
    weight = weight / torch.sum(weight, dim=1, keepdim=True)
    weight = weight.unsqueeze(1)
    index = index.view(batch_size, -1)
    index = index.unsqueeze(1).expand(-1, in_channels, -1)
    x = torch.gather(x, 2, index)
    x = x.view(batch_size, in_channels, k, -1)
    x = torch.sum(x * weight, dim=2)
    return x


def chamfer_loss(x, y):
    sqdist = cdist(x, y)
    return torch.mean(sqdist.min(1)[0]) + torch.mean(sqdist.min(2)[0])


def kernel_density(p, bandwidth, kernel="gaussian"):
    sqdist = cdist(p, p)
    var = bandwidth ** 2
    scale = bandwidth * math.sqrt(2 * math.pi)
    prob = torch.exp(-sqdist / (2 * var)) / scale
    density = torch.sum(prob, dim=2)
    return density
