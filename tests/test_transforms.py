import torch
import numpy as np
import torch3d.transforms as T


def test_to_tensor():
    size = torch.Size([3, 2048])
    x = np.random.rand(2048, 3)
    x = T.ToTensor()(x)
    assert x.is_contiguous()
    assert x.shape == size


def test_compose():
    size = torch.Size([3, 1024])
    x = np.random.rand(2048, 3)
    t = T.Compose([T.Shuffle(), T.RandomPointSample(1024), T.ToTensor()])
    x = t(x)
    assert x.is_contiguous()
    assert x.shape == size
