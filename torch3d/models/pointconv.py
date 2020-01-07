import torch
import torch.nn as nn
from torch3d.nn import PointConv


class PointConvNet(nn.Module):
    def __init__(self, in_channels, num_classes, dropout=0.5):
        super(PointConvNet, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.dropout = dropout
        self.conv1 = PointConv(in_channels, [64, 64, 128], 32, 4, 0.1, bias=False)
        self.conv2 = PointConv(128 + 3, [128, 128, 256], 64, 4, 0.2, bias=False)
        self.conv3 = PointConv(256 + 3, [256, 512, 1024], None, None, 0.4, bias=False)
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
