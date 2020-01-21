import torch
import torch.nn as nn
from torch3d.nn import SetConv


class PointNetSSG(nn.Module):
    """
    PointNet++ single-scale grouping model from the
    `"PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space" <https://arxiv.org/abs/1706.02413>`_ paper.

    Args:
        in_channels (int): Number of channels in the input point set
        num_classes (int): Number of classes in the dataset
        dropout (float, optional): Dropout rate in the classifier. Default: 0.5
    """  # noqa

    def __init__(self, in_channels, num_classes, dropout=0.5):
        super(PointNetSSG, self).__init__()
        self.conv1 = SetConv(in_channels, [64, 64, 128], 512, 32, 0.2, bias=False)
        self.conv2 = SetConv(128 + 3, [128, 128, 256], 128, 64, 0.4, bias=False)
        self.conv3 = SetConv(256 + 3, [256, 512, 1024], 1, 128, 0.8, bias=False)
        self.mlp = nn.Sequential(
            nn.Linear(1024, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.Dropout(dropout),
            nn.Linear(512, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            nn.Dropout(dropout),
        )
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.squeeze(2)
        x = self.mlp(x)
        x = self.fc(x)
        return x
