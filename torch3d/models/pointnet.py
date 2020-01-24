import torch
import torch.nn as nn


class PointNet(nn.Module):
    """
    PointNet classification architecture from the
    `"PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation" <https://arxiv.org/abs/1612.00593>`_ paper.

    Args:
        in_channels (int): Number of channels in the input point set
        num_classes (int): Number of classes in the dataset
        dropout (float, optional): Dropout rate in the classifier. Default: 0.5
    """  # noqa

    def __init__(self, in_channels, num_classes, dropout=0.5):
        super(PointNet, self).__init__()
        self.mlp1 = nn.Sequential(
            nn.Conv1d(in_channels, 64, 1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(True),
            nn.Conv1d(64, 64, 1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(True),
        )
        self.mlp2 = nn.Sequential(
            nn.Conv1d(64, 128, 1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(True),
            nn.Conv1d(128, 1024, 1, bias=False),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
        )
        self.maxpool = nn.AdaptiveMaxPool1d(1)
        self.mlp3 = nn.Sequential(
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
        x = self.mlp1(x)
        x = self.mlp2(x)
        x = self.maxpool(x).squeeze(2)
        x = self.mlp3(x)
        x = self.fc(x)
        return x
