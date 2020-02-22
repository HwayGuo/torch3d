import os
import h5py
import numpy as np
import torch.utils.data as data
from torchvision.datasets.utils import download_and_extract_archive, check_integrity


class ModelNet40(data.Dataset):
    """
    The `ModelNet40 <https://modelnet.cs.princeton.edu/>`_ dataset.

    Args:
        root (string): Root directory of dataset where the directory ``modelnet40`` exists or will be saved to if download is set to True.
        train (bool, optional): If True, create dataset from train set, otherwise create from test set. Default: ``True``
        transforms (callable, optional): A function/transform that takes input sample and its target as entry and return a transformed version. Default: ``None``
        download (bool, optional): If True, download the dataset and put it in the root directory. If the dataset is already downloaded, then do nothing. Default: ``False``
    """  # noqa

    name = "modelnet40"
    basedir = "modelnet40_ply_hdf5_2048"
    url = "https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip"

    splits = {
        "train": [
            ("ply_data_train0.h5", "3176385ffc31a7b6b5af22191fd920d1"),
            ("ply_data_train1.h5", "e3f613fb500559403b34925112754dc4"),
            ("ply_data_train2.h5", "0c56e233a090ff87c3049d4ce08e7d8b"),
            ("ply_data_train3.h5", "9d2af465adfa33a3285c369f3ca66c45"),
            ("ply_data_train4.h5", "dff38de489b2c41bfaeded86c2208984"),
        ],
        "test": [
            ("ply_data_test0.h5", "e9732e6d83b09e79e9a7617df058adee"),
            ("ply_data_test1.h5", "aba4b12a67c34391cc3c015a6f08ed4b"),
        ],
    }

    def __init__(self, root, train=True, transforms=None, download=False):
        super(ModelNet40, self).__init__()
        self.root = root
        self.train = train
        self.transforms = transforms

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError("Dataset not found or corrupted.")

        if self.train:
            filelist = self.splits["train"]
        else:
            filelist = self.splits["test"]

        self.data = []
        self.labels = []
        for filename, md5 in filelist:
            h5 = h5py.File(os.path.join(root, self.basedir, filename), "r")
            assert "data" in h5 and "label" in h5
            self.data.append(np.array(h5["data"][:]))
            self.labels.append(np.array(h5["label"][:]))
            h5.close()

        self.data = np.concatenate(self.data, 0)
        self.labels = np.concatenate(self.labels, 0)
        self.labels = np.squeeze(self.labels).astype(np.int64)

        with open(os.path.join(root, self.basedir, "shape_names.txt"), "r") as fp:
            self.categories = [x.strip() for x in fp]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        pcd = self.data[i]
        label = self.labels[i]
        if self.transforms is not None:
            pcd, label = self.transforms(pcd, label)
        return pcd, label

    def download(self):
        if self._check_integrity():
            print("Files already downloaded and verified")
            return
        download_and_extract_archive(self.url, self.root)
        os.rename(
            os.path.join(self.root, self.basedir), os.path.join(self.root, self.name)
        )

    def _check_integrity(self):
        flist = self.splits["train"] + self.splits["test"]
        for filename, md5 in flist:
            fpath = os.path.join(self.root, self.basedir, filename)
            if not check_integrity(fpath, md5):
                return False
        return True
