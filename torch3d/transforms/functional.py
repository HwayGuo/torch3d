import torch
import numpy as np


def _is_numpy(pcd):
    return isinstance(pcd, np.ndarray)


def _is_tensor(tensor):
    return isinstance(tensor, torch.Tensor)


def _is_numpy_point_cloud(pcd):
    return pcd.ndim == 2


def to_tensor(pcd):
    """
    Convert a ``numpy.ndarray`` point cloud to tensor.

    See ``ToTensor`` for more details.

    Args:
        pcd (numpy.ndarray): Point cloud to be converted to tensor.

    Returns:
        Tensor: Converted point cloud.
    """

    if not _is_numpy(pcd):
        raise TypeError("pcd should be ndarray. Got {}.".format(type(pcd)))

    if _is_numpy(pcd) and not _is_numpy_point_cloud(pcd):
        raise ValueError(
            "pcd should be 2 dimensional. Got {} dimensions.".format(pcd.ndim)
        )

    if _is_numpy(pcd):
        pcd = torch.from_numpy(pcd.T)
        return pcd


def to_numpy(tensor):
    """
    Converts a `Tensor` to a `numpy.ndarray` point cloud.
    """
    pass
