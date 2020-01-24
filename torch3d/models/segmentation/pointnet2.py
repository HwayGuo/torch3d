import torch
import torch.nn as nn
from torch3d.nn import SetAbstraction, FeaturePropagation


class PointNetSSG(nn.Module):
    def __init__(self, in_channels, num_classes, dropout=0.5):
        super(PointNetSSG, self).__init__()
        self.sa1 = SetAbstraction(in_channels, [32, 32, 64], 1024, 32, 0.1, bias=False)
        self.sa2 = SetAbstraction(64 + 3, [64, 64, 128], 256, 32, 0.2, bias=False)
        self.sa3 = SetAbstraction(128 + 3, [128, 128, 256], 64, 32, 0.4, bias=False)
        self.sa4 = SetAbstraction(256 + 3, [256, 256, 512], 16, 32, 0.8, bias=False)
        self.fp1 = FeaturePropagation(512 + 256, [256, 256], 3, bias=False)
        self.fp2 = FeaturePropagation(256 + 128, [256, 256], 3, bias=False)
        self.fp3 = FeaturePropagation(256 + 64, [256, 128], 3, bias=False)
        self.fp4 = FeaturePropagation(128, [128, 128], 3, bias=False)
        self.mlp = nn.Sequential(
            nn.Conv1d(128, 128, 1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(True),
            nn.Dropout(dropout),
            nn.Conv1d(128, 128, 1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(True),
            nn.Dropout(dropout),
        )
        self.fc = nn.Conv1d(128, num_classes, 1)

    def forward(self, x):
        p = x[:, :3]
        x1 = self.sa1(x)
        x2 = self.sa2(x1)
        x3 = self.sa3(x2)
        x4 = self.sa4(x3)
        x3 = self.fp1(x4, x3)
        x2 = self.fp2(x3, x2)
        x1 = self.fp3(x2, x1)
        x = self.fp4(x1, p)
        x = x[:, 3:]
        x = self.mlp(x)
        x = self.fc(x)
        return x
