import os
import h5py
import numpy as np
from torchvision.datasets import VisionDataset
from torchvision.datasets.utils import download_file_from_google_drive, extract_archive


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

    def download(self):
        if not self._check_integrity():
            download_file_from_google_drive(self.fileid, self.root, self.filename)
            extract_archive(
                os.path.join(self.root, self.filename),
                os.path.join(self.root, self.name),
            )

    def _check_integrity(self):
        return False


if __name__ == "__main__":
    dataset = S3DIS("data", download=True)
