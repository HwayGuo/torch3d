import torch
import torch.nn as nn
from torch3d.nn import SetConv


class PointNetSSG(nn.Module):
    def __init__(self, in_channels, num_classes, dropout=0.5):
        super(PointNetSSG, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.dropout = dropout
        self.conv1 = SetConv(in_channels, [32, 32, 64], 32, 4, 0.1, bias=False)
        self.conv2 = SetConv(64 + 3, [64, 64, 128], 32, 4, 0.2, bias=False)
        self.conv3 = SetConv(128 + 3, [128, 128, 256], 32, 4, 0.4, bias=False)
        self.conv4 = SetConv(256 + 3, [256, 256, 512], 32, 4, 0.8, bias=False)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        return x


if __name__ == "__main__":
    x = torch.rand(2, 3, 1024)
    m = PointNetSSG(3, 40).cuda()
    m(x.cuda())
