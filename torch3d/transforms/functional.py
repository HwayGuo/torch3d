import torch
import numpy as np


def _is_numpy(points):
    return isinstance(points, np.ndarray)


def _is_tensor(tensor):
    return isinstance(tensor, torch.Tensor)


def _is_numpy_point_cloud(points):
    return points.ndim == 2


def to_tensor(points):
    """Convert a point cloud to tensor.

    See ``ToTensor`` for more details.

    Args:
      points (numpy.ndarray): Point cloud to be converted to tensor.

    Returns:
      torch.Tensor: Converted point cloud.
    """

    if not _is_numpy(points):
        raise TypeError(
            "Point cloud should be an ndarray. Got {}.".format(type(points))
        )

    if _is_numpy(points) and not _is_numpy_point_cloud(points):
        raise ValueError(
            "Point cloud should be 2 dimensional. Got {} dimensions.".format(
                points.ndim
            )
        )

    if _is_numpy(points):
        points = torch.as_tensor(points).contiguous()
        return points
