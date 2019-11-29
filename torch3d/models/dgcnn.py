import torch
import torch.nn as nn
from torch3d.nn import EdgeConv


__all__ = ["DGCNN"]


class DGCNN(nn.Module):
    def __init__(self, in_channels, num_classes, dropout=0.5):
        super(DGCNN, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.dropout = dropout
        self.conv1 = EdgeConv(self.in_channels, 64, 20, bias=False)
        self.conv2 = EdgeConv(64, 64, 20, bias=False)
        self.conv3 = EdgeConv(64, 128, 20, bias=False)
        self.conv4 = EdgeConv(128, 256, 20, bias=False)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.maxpool = nn.AdaptiveMaxPool1d(1)
        self.mlp = nn.Sequential(
            nn.Linear(1024, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.Dropout(self.dropout),
            nn.Linear(512, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            nn.Dropout(self.dropout),
        )
        self.fc = nn.Linear(256, self.num_classes)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x = torch.cat([x1, x2, x3, x4], dim=1)
        x = torch.cat([self.avgpool(x), self.maxpool(x)], dim=1).squeeze(-1)
        x = self.mlp(x)
        x = self.fc(x)
        return x
