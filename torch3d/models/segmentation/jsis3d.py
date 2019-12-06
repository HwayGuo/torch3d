import torch
import torch.nn as nn


__all__ = ["JSIS3D"]


class JSIS3D(nn.Module):
    def __init__(self, backbone, num_classes, embedding_size):
        super(JSIS3D, self).__init__()
        self.num_classes = num_classes
        self.embedding_size = embedding_size
        self.fan = backbone.fc.weight.data.shape[1]
        self.net = backbone
        self.net.fc = JSIS3DHead(self.fan, self.num_classes, self.embedding_size)

    def forward(self, x):
        return self.net(x)


class JSIS3DHead(nn.Module):
    def __init__(self, in_channels, num_classes, embedding_size):
        super(JSIS3DHead, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.embedding_size = embedding_size
        self.fc1 = nn.Conv1d(self.in_channels, self.num_classes, 1)
        self.fc2 = nn.Conv1d(self.in_channels, self.embedding_size, 1)

    def forward(self, x):
        x1 = self.fc1(x)
        x2 = self.fc2(x)
        return x1, x2
