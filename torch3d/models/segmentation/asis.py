import torch
import torch.nn as nn
from torch3d.nn import SetAbstraction, FeaturePropagation, FarthestPointSample


__all__ = ["ASIS", "ASISHead"]


class ASIS(nn.Module):
    def __init__(self, in_channels, num_classes, embedding_size, dropout=0.5):
        super(ASIS, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.embedding_size = embedding_size
        self.dropout = dropout
        self.down1 = FarthestPointSample(1024)
        self.down2 = FarthestPointSample(256)
        self.down3 = FarthestPointSample(64)
        self.down4 = FarthestPointSample(16)
        self.sa1 = SetAbstraction(in_channels + 3, [32, 32, 64], 0.1, 32, bias=False)
        self.sa2 = SetAbstraction(64 + 3, [64, 64, 128], 0.2, 32, bias=False)
        self.sa3 = SetAbstraction(128 + 3, [128, 128, 256], 0.4, 32, bias=False)
        self.sa4 = SetAbstraction(256 + 3, [256, 256, 512], 0.8, 32, bias=False)
        self.fp11 = FeaturePropagation(768, [256, 256], bias=False)
        self.fp12 = FeaturePropagation(384, [256, 256], bias=False)
        self.fp13 = FeaturePropagation(320, [256, 128], bias=False)
        self.fp14 = FeaturePropagation(128, [128, 128], bias=False)
        self.fp21 = FeaturePropagation(768, [256, 256], bias=False)
        self.fp22 = FeaturePropagation(384, [256, 256], bias=False)
        self.fp23 = FeaturePropagation(320, [256, 128], bias=False)
        self.fp24 = FeaturePropagation(128, [128, 128], bias=False)


class ASISHead(nn.Module):
    pass
