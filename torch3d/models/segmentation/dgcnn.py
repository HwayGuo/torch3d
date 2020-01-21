import torch
import torch.nn as nn
from torch3d.nn import EdgeConv


class DGCNN(nn.Module):
    def __init__(self, in_channels, num_classes, dropout=0.5):
        super(DGCNN, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.dropout = dropout
        self.conv1 = EdgeConv(in_channels, [64, 64], 20, bias=False)
        self.conv2 = EdgeConv(64, [64, 64], 20, bias=False)
        self.conv3 = EdgeConv(64, 64, 20, bias=False)
        self.conv4 = nn.Sequential(
            nn.Conv1d(192, 1024, 1, bias=False),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2, True),
        )
        self.maxpool = nn.AdaptiveMaxPool1d(1)
        self.mlp = nn.Sequential(
            nn.Conv1d(1216, 512, 1, bias=False),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, True),
            nn.Conv1d(512, 256, 1, bias=False),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, True),
            nn.Dropout(dropout),
            nn.Conv1d(256, 128, 1, bias=False),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2, True),
            nn.Dropout(dropout),
        )
        self.fc = nn.Conv1d(128, num_classes, 1)

    def forward(self, x):
        num_points = x.shape[2]
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x = torch.cat([x1, x2, x3], dim=1)
        x = self.conv4(x)
        x = self.maxpool(x).repeat(1, 1, num_points)
        x = torch.cat([x, x1, x2, x3], dim=1)
        x = self.mlp(x)
        x = self.fc(x)
        return x
