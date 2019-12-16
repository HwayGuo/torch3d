import torch
import numpy as np
from torch3d.transforms import functional as F


class Compose(object):
    """
    Composes several transforms together.

    Args:
        transforms (list of ``Transform``): list of transforms to compose.
    Example:
        >>> transforms.Compose([
        >>>     transforms.Shuffle(),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, pcd):
        for t in self.transforms:
            pcd = t(pcd)
        return pcd


class ToTensor(object):
    def __call__(self, pcd):
        return F.to_tensor(pcd)


class Shuffle(object):
    @staticmethod
    def get_params(pcd):
        n = len(pcd)
        assert n > 0
        return np.random.permutation(n)

    def __call__(self, pcd):
        index = self.get_params(pcd)
        return pcd[index]


class RandomPointSample(object):
    def __init__(self, num_samples):
        self.num_samples = num_samples

    @staticmethod
    def get_params(pcd, num_samples):
        n = len(pcd)
        assert n > 0
        if n >= num_samples:
            index = np.random.choice(n, num_samples, replace=False)
        else:
            m = num_samples - n
            index = np.random.choice(n, m, replace=True)
            index = list(range(n)) + list(index)
        return index

    def __call__(self, pcd):
        index = self.get_params(pcd, self.num_samples)
        return pcd[index]
