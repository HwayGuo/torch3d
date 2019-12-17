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
