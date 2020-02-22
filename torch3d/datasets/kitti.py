import os
import h5py
import numpy as np
import torch.utils.data as data


class KITTI(data.Dataset):
    """
    The `KITTI <http://www.cvlibs.net/datasets/kitti/>`_ dataset.

    Args:
        root (string): Root directory of dataset where the directory ``kitti`` exists.
        train (bool, optional): If ``True``, create dataset from train
    """  # noqa

    name = "kitti"

    def __init__(self, root, split="train", transforms=None):
        super(KITTI, self).__init__()
        self.root = root
        self.split = split
        self.transforms = transforms

        if not self._check_integrity():
            raise RuntimeError("Dataset not found or corrupted.")
