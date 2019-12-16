import torch
import torch.nn.functional as F
from torch3d.extension import _lazy_import


def cdist(x, y):
    xx = x.pow(2).sum(dim=1, keepdim=True).permute(0, 2, 1)
    yy = y.pow(2).sum(dim=1, keepdim=True)
    sqdist = torch.baddbmm(yy, x.permute(0, 2, 1), y, alpha=-2).add_(xx)
    return sqdist


def knn(p, q, k):
    sqdist = cdist(q, p)
    return torch.topk(sqdist, k, dim=2, largest=False)


def ball_point(p, q, k, radius):
    _C = _lazy_import()
    return _C.ball_point(p, q, k, radius)


def random_point_sample(p, num_samples):
    num_points = p.shape[1]
    if num_samples > num_points:
        raise ValueError("num_samples should be less than input size.")
    return torch.randperm(num_points)[:num_samples]


def farthest_point_sample(p, num_samples):
    num_points = p.shape[1]
    if num_samples > num_points:
        raise ValueError("num_samples should be less than input size.")
    _C = _lazy_import()
    return _C.farthest_point_sample(p, num_samples)


def batched_index_select(x, dim, index):
    views = [x.shape[0]]
    views += [1 if i != dim else -1 for i in range(1, len(x.shape))]
    expanse = list(x.shape)
    expanse[dim] = -1
    index = index.view(views).expand(expanse)
    return torch.gather(x, dim, index)
