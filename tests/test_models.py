import torch
import torch3d.models as models


class TestPointNet:
    batch_size = 8
    num_points = 2048
    in_channels = 3
    num_classes = 100
    model = models.PointNet(in_channels, num_classes)

    def test_forward(self):
        self.model.eval()
        x = torch.rand(self.batch_size, self.in_channels, self.num_points)
        y = self.model(x)
        assert y.shape == torch.Size([self.batch_size, self.num_classes])


class TestPointNetSegmentation:
    batch_size = 8
    num_points = 2048
    in_channels = 3
    num_classes = 100
    model = models.segmentation.PointNet(in_channels, num_classes)

    def test_forward(self):
        self.model.eval()
        x = torch.rand(self.batch_size, self.in_channels, self.num_points)
        y = self.model(x)
        assert y.shape == torch.Size(
            [self.batch_size, self.num_classes, self.num_points]
        )


class TestPointNet2:
    batch_size = 8
    num_points = 2048
    in_channels = 0
    num_classes = 100
    model = models.PointNetSSG(in_channels, num_classes)

    def test_forward(self):
        self.model.cuda()
        self.model.eval()
        x = torch.rand(self.batch_size, self.num_points, 3).cuda()
        y = self.model(x)
        assert y.shape == torch.Size([self.batch_size, self.num_classes])


class TestPointCNN:
    batch_size = 8
    num_points = 2048
    in_channels = 0
    num_classes = 100
    model = models.PointCNN(in_channels, num_classes)

    def test_forward(self):
        self.model.eval()
        x = torch.rand(self.batch_size, self.num_points, 3)
        y = self.model(x)
        assert y.shape == torch.Size([self.batch_size, self.num_classes])


class TestDGCNN:
    batch_size = 8
    num_points = 2048
    in_channels = 3
    num_classes = 100
    model = models.DGCNN(in_channels, num_classes)

    def test_forward(self):
        self.model.eval()
        x = torch.rand(self.batch_size, self.in_channels, self.num_points)
        y = self.model(x)
        assert y.shape == torch.Size([self.batch_size, self.num_classes])


class TestDGCNNSegmentation:
    batch_size = 8
    num_points = 2048
    in_channels = 3
    num_classes = 100
    model = models.segmentation.DGCNN(in_channels, num_classes)

    def test_forward(self):
        self.model.eval()
        x = torch.rand(self.batch_size, self.in_channels, self.num_points)
        y = self.model(x)
        assert y.shape == torch.Size(
            [self.batch_size, self.num_classes, self.num_points]
        )


class TestJSIS3D:
    batch_size = 8
    num_points = 2048
    in_channels = 3
    num_classes = 100
    embedding_size = 32
    model = models.segmentation.JSIS3D(in_channels, num_classes, embedding_size)

    def test_forward(self):
        self.model.eval()
        x = torch.rand(self.batch_size, self.in_channels, self.num_points)
        y = self.model(x)
        assert y[0].shape == torch.Size(
            [self.batch_size, self.num_classes, self.num_points]
        )
        assert y[1].shape == torch.Size(
            [self.batch_size, self.embedding_size, self.num_points]
        )
