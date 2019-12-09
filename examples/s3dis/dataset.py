import os
import h5py
import numpy as np
from torchvision.datasets import VisionDataset
from torchvision.datasets.utils import (
    download_file_from_google_drive,
    extract_archive,
    list_files,
)


class S3DIS(VisionDataset):
    name = "s3dis"
    fileid = "1pZHCRJpUvReQYIcR16SISnGZl6Nw-Bfr"
    filename = "s3dis_h5_07042019.zip"

    def __init__(
        self,
        root,
        train=True,
        test_area=5,
        download=False,
        transform=None,
        target_transform=None,
        transforms=None,
    ):
        super(S3DIS, self).__init__(root, transforms, transform, target_transform)
        self.train = train
        self.test_area = test_area

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError("Dataset not found or corrupted.")

        flist = list_files(os.path.join(self.root, self.name), ".h5")
        # Filter files that is not in the area of interest
        area = "Area_" + str(self.test_area)
        index = [i for i, filename in enumerate(flist) if area in filename]
        if self.train:
            index = list(set(range(len(flist))) - set(index))
        flist = [flist[i] for i in index]

        self.data = []
        self.targets = []

        for filename in flist:
            h5 = h5py.File(os.path.join(self.root, self.name, filename), "r")
            assert "points" in h5 and "labels" in h5
            self.data.append(np.array(h5["points"][:]))
            self.targets.append(np.array(h5["labels"][:]))
            h5.close()
        self.data = np.concatenate(self.data, axis=0)
        self.targets = np.concatenate(self.targets, axis=0)

        for i in range(len(self.targets)):
            self.targets[i, :, 1] = np.unique(self.targets[i, :, 1], False, True)[1]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        pcd = self.data[i]
        target = self.targets[i]
        if self.transforms is not None:
            pcd, target = self.transforms(pcd, target)
        return pcd, target

    def download(self):
        if not self._check_integrity():
            download_file_from_google_drive(self.fileid, self.root, self.filename)
            extract_archive(
                os.path.join(self.root, self.filename),
                os.path.join(self.root, self.name),
            )

    def _check_integrity(self):
        flist = list_files(os.path.join(self.root, self.name), ".h5")
        return flist
