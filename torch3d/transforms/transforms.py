import torch
import numpy as np
import torch3d.transforms.functional as F


class Compose(object):
    """Compose several transforms together.

    Args:
      transforms (list[callable]): List of transforms to compose.

    Example:
      >>> transforms.Compose([
      >>>     transforms.Shuffle(),
      >>>     transforms.ToTensor(),
      >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, points):
        for t in self.transforms:
            points = t(points)
        return points


class ToTensor(object):
    """Convert a numpy.ndarray to tensor."""

    def __call__(self, points):
        """
        Args:
          points (numpy.ndarray): Point cloud to be converted to tensor.

        Returns:
          torch.Tensor: Converted point cloud.
        """
        return F.to_tensor(points)

    def __repr__(self):
        return self.__class__.__name__ + "()"


class Shuffle(object):
    """Randomly shuffle the given point cloud."""

    @staticmethod
    def get_params(points):
        """Get a random order for a shuffle.

        Args:
          points (numpy.ndarray): Point cloud to be shuffled.

        Returns:
          numpy.ndarray: A random permuted sequence.
        """
        n = len(points)
        assert n > 0
        return np.random.permutation(n)

    def __call__(self, points):
        perm = self.get_params(points)
        return points[perm]


class RandomPointSample(object):
    """Randomly sample the given point cloud.

    Args:
      num_samples (int): Number of samples to select from point cloud.
    """

    def __init__(self, num_samples):
        self.num_samples = num_samples

    @staticmethod
    def get_params(points, num_samples):
        n = len(points)
        assert n > 0
        if n >= num_samples:
            samples = np.random.choice(n, num_samples, replace=False)
        else:
            m = num_samples - n
            samples = np.random.choice(n, m, replace=True)
            samples = list(range(n)) + list(samples)
        return samples

    def __call__(self, points):
        samples = self.get_params(points, self.num_samples)
        return points[samples]
