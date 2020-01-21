import torch
import torch.nn as nn
from torch3d.nn import SetConv, SetConvTranspose


class PointNetSSG(nn.Module):
    def __init__(self, in_channels, num_classes, dropout=0.5):
        super(PointNetSSG, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.dropout = dropout
        self.conv1 = SetConv(self.in_channels, [32, 32, 64], 32, 4, 0.1, bias=False)
        self.conv2 = SetConv(64 + 3, [64, 64, 128], 32, 4, 0.2, bias=False)
        self.conv3 = SetConv(128 + 3, [128, 128, 256], 32, 4, 0.4, bias=False)
        self.conv4 = SetConv(256 + 3, [256, 256, 512], 32, 4, 0.8, bias=False)
        self.dconv1 = SetConvTranspose(512 + 256, [256, 256], 3, bias=False)
        self.dconv2 = SetConvTranspose(256 + 128, [256, 256], 3, bias=False)
        self.dconv3 = SetConvTranspose(256 + 64, [256, 128], 3, bias=False)
        self.dconv4 = SetConvTranspose(128, [128, 128], 3, bias=False)
        self.mlp = nn.Sequential(
            nn.Conv1d(128, 128, 1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(True),
            nn.Dropout(self.dropout),
            nn.Conv1d(128, 128, 1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(True),
            nn.Dropout(self.dropout),
        )
        self.fc = nn.Conv1d(128, self.num_classes, 1)

    def forward(self, x):
        p = x[:, :3]
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x3 = self.dconv1(x4, x3)
        x2 = self.dconv2(x3, x2)
        x1 = self.dconv3(x2, x1)
        x = self.dconv4(x1, p)
        x = x[:, 3:]
        x = self.mlp(x)
        x = self.fc(x)
        return x
