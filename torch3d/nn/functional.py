import torch
import torch.nn.functional as F
from torch3d.extension import _lazy_import


def cdist(x, y):
    xx = x.pow(2).sum(dim=2, keepdim=True)
    yy = y.pow(2).sum(dim=2, keepdim=True).transpose(2, 1)
    sqdist = torch.baddbmm(yy, x, y.transpose(2, 1), alpha=-2).add_(xx)
    return sqdist


def knn(p, q, k):
    sqdist = cdist(q, p)
    return torch.topk(sqdist, k, dim=-1, largest=False)


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


def point_interpolate(x, index, weight):
    return PointInterpolate.apply(x, index, weight)


class PointInterpolate(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, index, weight):
        ctx.num_points = x.shape[2]
        ctx.save_for_backward(index, weight)
        _C = _lazy_import()
        return _C.point_interpolate(x, index, weight)

    @staticmethod
    def backward(ctx, grad):
        num_points = ctx.num_points
        index, weight = ctx.saved_tensors
        _C = _lazy_import()
        output = _C.point_interpolate_grad(grad, index, weight, num_points)
        return output, None, None
