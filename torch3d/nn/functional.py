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


def ball_point(p, q, radius, k):
    _C = _lazy_import()
    return _C.ball_point(p, q, radius, k)


def chamfer_loss(x, y):
    sqdist = cdist(x, y)
    return torch.mean(sqdist.min(1)[0]) + torch.mean(sqdist.min(2)[0])


def discriminative_loss(
    x, y, alpha=1.0, beta=1.0, gamma=0.001, delta_v=0.5, delta_d=1.5, reduction="mean"
):
    # TODO: Adhere to reduction rule
    batch_size = x.shape[0]
    channels = x.shape[1]
    num_points = x.shape[2]
    device = x.device

    sizes = y.max(1)[0] + 1  # number of instances in each sample
    y = F.one_hot(y)
    K = y.shape[2]

    x = x.unsqueeze(3).expand(-1, -1, -1, K)
    y = y.unsqueeze(1)
    x = x * y
    mu = torch.zeros(batch_size, channels, K).to(device)
    for i in range(batch_size):
        n = sizes[i]
        mu[i, :, :n] = x[i, :, :, :n].sum(1) / y[i, :, :, :n].sum(1)

    # Calculate variance term
    var = torch.norm(
        x - mu.unsqueeze(2).expand(-1, -1, num_points, -1), 2, dim=1, keepdim=True
    )
    var = torch.clamp(var - delta_v, min=0.0) ** 2
    var = var * y
    loss_v = 0
    for i in range(batch_size):
        n = sizes[i]
        v = var[i, :, :, :n].sum(1) / y[i, :, :, :n].sum(1)
        loss_v += v.sum() / n
    loss_v /= batch_size

    # Calculate inter-distance term
    loss_d = 0
    for i in range(batch_size):
        n = sizes[i]
        if n <= 1:
            continue
        margin = 2 * delta_d * (1.0 - torch.eye(n)).to(device)
        mu_a = mu[i, :, :n].unsqueeze(2).expand(-1, n, n)
        mu_b = mu_a.permute(0, 2, 1)
        dist = torch.norm(mu_a - mu_b, 2, dim=0)
        dist = torch.sum(torch.clamp(margin - dist, min=0.0) ** 2)
        dist /= float(n * (n - 1))
        loss_d += dist
    loss_d /= batch_size

    # Calculate regularization term
    loss_r = 0
    for i in range(batch_size):
        n = sizes[i]
        loss_r += torch.mean(torch.norm(mu[i, :, :n], 2, dim=0))
    loss_r /= batch_size

    loss = alpha * loss_v + beta * loss_d + gamma * loss_r
    return loss


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
