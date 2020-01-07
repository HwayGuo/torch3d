import torch
import torch3d


def test_meshgrid2d():
    start = 0.0
    end = 1.0
    steps = 2
    g = torch3d.meshgrid2d(start, end, steps)
    size = torch.Size([2, steps ** 2])
    assert g.shape == size
    assert g.tolist() == [[0.0, 0.0, 1.0, 1.0], [0.0, 1.0, 0.0, 1.0]]
