import os
import h5py
import numpy as np
import torch.utils.data as data
from torchvision.datasets.utils import download_and_extract_archive, check_integrity
from collections import OrderedDict


class ShapeNetPart(data.Dataset):
    """
    The `ShapeNet part segmentation <https://shapenet.cs.stanford.edu/iccv17/>`_ dataset.

    Args:
        root (string): Root directory of dataset where the directory ``hdf5_data``
            exists or will be saved to if download is set to True.
        train (bool, optional): If True, create dataset from train set, otherwise create from
            test set. Default: ``True``
        transforms (callable, optional): A function/transform that takes input sample and its
            target as entry and return a transformed version. Default: ``None``
        download (bool, optional): If True, download the dataset and put it in the root directory.
            If the dataset is already downloaded, then do nothing. Default: ``False``
    """

    url = "https://shapenet.cs.stanford.edu/media/shapenet_part_seg_hdf5_data.zip"
    basedir = "hdf5_data"
    splits = {
        "train": [
            ("ply_data_train0.h5", "e7152ff588ae6c87eb1156823df05855"),
            ("ply_data_train1.h5", "b4894a082211418b61b5a707fedb4f56"),
            ("ply_data_train2.h5", "508eeeee96053b90388520c37df3a8b8"),
            ("ply_data_train3.h5", "88574c3d5c61d0f3156b9e02cd6cda03"),
            ("ply_data_train4.h5", "418cb01104740bf1353b792331cb5878"),
            ("ply_data_train5.h5", "26e65c8827b08f7c340cd03f902e27e8"),
        ],
        "val": [("ply_data_val0.h5", "628b4b3cbc17765de2114d104e51b9c9")],
        "test": [
            ("ply_data_test0.h5", "fa3fb32b179128ede32c2c948ed83efc"),
            ("ply_data_test1.h5", "5eb63ae378831c665282c8f22b6c1249"),
        ],
    }

    categories = OrderedDict(
        [
            ("airplane", "02691156"),
            ("bag", "02773838"),
            ("cap", "02954340"),
            ("car", "02958343"),
            ("chair", "03001627"),
            ("earphone", "03261776"),
            ("guitar", "03467517"),
            ("knife", "03624134"),
            ("lamp", "03636649"),
            ("laptop", "03642806"),
            ("motorbike", "03790512"),
            ("mug", "03797390"),
            ("pistol", "03948459"),
            ("rocket", "04099429"),
            ("skateboard", "04225987"),
            ("table", "04379243"),
        ]
    )

    def __init__(self, root, split="train", transforms=None, download=False):
        super(ShapeNetPart, self).__init__()
        self.root = root
        self.split = split
        self.transforms = transforms

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError("Dataset not found or corrupted.")

        filelist = self.splits[split]

        self.data = []
        self.labels = []
        self.parts = []
        for filename, md5 in filelist:
            h5 = h5py.File(os.path.join(self.root, self.basedir, filename), "r")
            assert "data" in h5 and "label" in h5 and "pid" in h5
            self.data.append(np.array(h5["data"][:]))
            self.labels.append(np.array(h5["label"][:]))
            self.parts.append(np.array(h5["pid"][:]))
            h5.close()

        self.data = np.concatenate(self.data, 0)
        self.labels = np.concatenate(self.labels, 0)
        self.labels = np.squeeze(self.labels).astype(np.int64)
        self.parts = np.concatenate(self.parts, 0).astype(np.int64)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        pcd = self.data[i]
        target = {"label": self.labels[i], "partseg": self.parts[i]}
        if self.transforms is not None:
            pcd, target = self.transforms(pcd, target)
        return pcd, target

    def download(self):
        if self._check_integrity():
            print("Files already downloaded and verified")
            return
        download_and_extract_archive(self.url, self.root)

    def _check_integrity(self):
        filelist = self.splits["train"] + self.splits["val"] + self.splits["test"]
        for filename, md5 in filelist:
            fpath = os.path.join(self.root, self.basedir, filename)
            if not check_integrity(fpath, md5):
                return False
        return True
