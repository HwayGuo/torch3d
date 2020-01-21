import torch
import torch.nn as nn
from torch3d.nn import XConv


class PointCNN(nn.Module):
    """
    PointCNN classification model from the
    `"PointCNN: Convolution On X-Transformed Points" <https://arxiv.org/abs/1801.07791>`_ paper.

    Args:
        in_channels (int): Number of channels in the input point set
        num_classes (int): Number of classes in the dataset
        dropout (float, optional): Dropout rate in the classifier. Default: 0.5
    """  # noqa

    def __init__(self, in_channels, num_classes, dropout=0.5):
        super(PointCNN, self).__init__()
        self.conv1 = XConv(in_channels, 48, 1024, 8, dilation=1, bias=False)
        self.conv2 = XConv(48 + 3, 96, 384, 12, dilation=2, bias=False)
        self.conv3 = XConv(96 + 3, 192, 128, 16, dilation=2, bias=False)
        self.conv4 = XConv(192 + 3, 384, 128, 16, dilation=3, bias=False)
        self.mlp = nn.Sequential(
            nn.Conv1d(384, 256, 1, bias=False),
            nn.BatchNorm1d(256),
            nn.ELU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(256, 128, 1, bias=False),
            nn.BatchNorm1d(128),
            nn.ELU(inplace=True),
            nn.Dropout(dropout),
        )
        self.fc = nn.Conv1d(128, num_classes, 1)
        self.avgpool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x[:, 3:]
        x = self.mlp(x)
        x = self.fc(x)
        x = self.avgpool(x).squeeze(2)
        return x
