import torch
import torch3d.models as models


def test_classification():
    names = ["PointNet", "PointNetSSG", "DGCNN", "PointConvNN"]
    batch_size = 2
    in_channels = 3
    num_points = 2048
    num_classes = 40
    x = torch.rand(batch_size, in_channels, num_points)
    size = torch.Size([batch_size, num_classes])

    for name in names:
        cls = getattr(models, name)
        model = cls(in_channels, num_classes)
        model.eval()
        assert model(x).shape == size


def test_segmentation():
    names = ["PointNet", "PointNetSSG", "DGCNN"]
    batch_size = 2
    in_channels = 3
    num_points = 4096
    num_classes = 40
    x = torch.rand(batch_size, in_channels, num_points)
    size = torch.Size([batch_size, num_classes, num_points])

    for name in names:
        cls = getattr(models.segmentation, name)
        model = cls(in_channels, num_classes)
        model.eval()
        assert model(x).shape == size
