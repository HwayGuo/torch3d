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
    """
    Convert a ``numpy.ndarray`` to tensor.

    Converts a numpy.ndarray (N x C) to a torch.FloatTensor of shape (C x N).
    """

    def __call__(self, pcd):
        """
        Args:
            pcd (numpy.ndarray): Point cloud to be converted to tensor.

        Returns:
            Tensor: Converted point cloud.
        """
        return F.to_tensor(pcd)

    def __repr__(self):
        return self.__class__.__name__ + "()"


class Shuffle(object):
    """
    Randomly shuffle the given point cloud.

    Shuffles a numpy.ndarray (N x C) by random permutation.
    """

    @staticmethod
    def get_params(pcd):
        """
        Get a random order for a shuffle.

        Args:
            pcd (numpy.ndarray): Point cloud to be shuffled.

        Returns:
            numpy.ndarray: A random permuted sequence.
        """

        n = len(pcd)
        assert n > 0
        return np.random.permutation(n)

    def __call__(self, pcd):
        perm = self.get_params(pcd)
        return pcd[perm]


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
