import torch
import torch.nn as nn
import torch3d.ops as ops


class PointPillars(nn.Module):
    """PointPillars detection architecture.

    Args:
      in_channels (int): Number of channels in the input point cloud.
      min_range (Tuple[float, float, float]):
      max_range (Tuple[float, float, float]):
      pillar_size (float or Tuple[float, float, float]):
      max_points_per_pillar (int):
      max_pillars (int)
    """

    def __init__(self):
        super(PointPillars, self).__init__()

    def forward(self, x):
        print(x.shape)
