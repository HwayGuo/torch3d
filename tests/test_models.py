import torch
import torch3d.models as models


def test_classification():
    names = ["PointNet", "DGCNN"]
    batch_size = 8
    in_channels = 3
    num_points = 2048
    num_classes = 100
    x = torch.rand(batch_size, in_channels, num_points)
    size = torch.Size([batch_size, num_classes])

    for name in names:
        cls = getattr(models, name)
        model = cls(in_channels, num_classes)
        model.eval()
        assert model(x).shape == size
