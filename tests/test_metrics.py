import torch
import torch3d.metrics as metrics


def test_accuracy():
    num_classes = 4
    metric = metrics.Accuracy(num_classes)
    x = torch.tensor([[0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
    y = torch.tensor([1, 2])
    metric.update(x, y)
    assert metric.score() == 0.5
    assert metric.mean() == 0.5


def test_iou():
    num_classes = 4
    metric = metrics.IoU(num_classes)
    x = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
    y = torch.tensor([1, 2])
    metric.update(x, y)
    assert metric.score() == 0.0
    assert metric.mean() == 0.0
