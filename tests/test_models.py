import torch
import torch3d.models as models


class TestClassification:
    batch_size = 8
    num_points = 2048
    in_channels = 3
    num_classes = 100
    points = torch.rand(batch_size, in_channels, num_points)
    output_shape = torch.Size([batch_size, num_classes])

    def test_pointnet(self):
        model = models.PointNet(self.in_channels, self.num_classes)
        model.eval()
        output = model(self.points)
        assert output.shape == self.output_shape

    def test_pointnet2(self):
        model = models.PointNetSSG(self.in_channels - 3, self.num_classes).cuda()
        model.eval()
        output = model(self.points.transpose(2, 1).cuda())
        assert output.shape == self.output_shape

    def test_pointcnn(self):
        model = models.PointCNN(self.in_channels - 3, self.num_classes)
        model.eval()
        output = model(self.points.transpose(2, 1))
        assert output.shape == self.output_shape

    def test_dgcnn(self):
        model = models.DGCNN(self.in_channels, self.num_classes)
        model.eval()
        output = model(self.points)
        assert output.shape == self.output_shape
