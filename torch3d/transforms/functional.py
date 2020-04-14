import torch
import numpy as np


def _is_numpy(points):
    return isinstance(points, np.ndarray)


def _is_tensor(tensor):
    return isinstance(tensor, torch.Tensor)


def _is_numpy_point_cloud(pcd):
    return pcd.ndim == 2


def to_tensor(points):
    """Convert a ``np.ndarray`` point cloud to tensor.

    See ``ToTensor`` for more details.

    Args:
      points (np.ndarray): Point cloud to be converted to tensor.

    Returns:
      Tensor: Converted point cloud.
    """

    if not _is_numpy(points):
        raise TypeError(
            "Point cloud should be an ndarray. Got {}.".format(type(points))
        )

    if _is_numpy(pcd) and not _is_numpy_point_cloud(pcd):
        raise ValueError(
            "Point cloud should be 2 dimensional. Got {} dimensions.".format(pcd.ndim)
        )

    if _is_numpy(points):
        points = torch.as_tensor(points.T).contiguous()
        return points


def to_numpy(tensor):
    """
    Converts a `Tensor` to a `numpy.ndarray` point cloud.
    """
    pass
