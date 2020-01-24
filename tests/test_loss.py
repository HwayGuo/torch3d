import torch
import torch3d.nn as nn
import torch3d.nn.functional as F


def test_chamfer_loss():
    x = torch.rand([1, 3, 2048])
    loss = nn.ChamferLoss()(x, x)
    zeros = torch.zeros_like(loss)
    assert torch.allclose(loss, zeros)
