import os
import re
import h5py
import numpy as np
import torch.utils.data as data
from PIL import Image


class KITTIDetection(data.Dataset):
    """
    The `KITTI <http://www.cvlibs.net/datasets/kitti/>`_ dataset.

    Args:
      root (string): Root directory of the KITTI dataset.
      transforms (callable, optional): A function/transform that takes input sample and
        its target as entry and return a transformed version. Default: ``None``
    """  # noqa

    def __init__(self, root, split="train", transforms=None):
        super(KITTIDetection, self).__init__()
        self.root = root
        if split in ["train", "val"]:
            self.split = "training"
        else:
            self.split = "testing"
        self.transforms = transforms

        self.image_path = os.path.join(root, self.split, "image_2")
        self.lidar_path = os.path.join(root, self.split, "velodyne")
        self.label_path = os.path.join(root, self.split, "label_2")
        self.calib_path = os.path.join(root, self.split, "calib")

        if not self._check_integrity():
            raise RuntimeError("Dataset not found or corrupted.")

        self.framelist = sorted(
            [
                int(x.split(".")[0])
                for x in os.listdir(self.lidar_path)
                if re.match("^[0-9]{6}.bin$", x)
            ]
        )

    def _check_integrity(self):
        return (
            os.path.exists(self.image_path)
            and os.path.exists(self.calib_path)
            and os.path.exists(self.lidar_path)
            and (os.path.exists(self.label_path) if self.split == "training" else True)
        )

    def __len__(self):
        return len(self.framelist)

    def __getitem__(self, i):
        frameid = self.framelist[i]
        image = self._get_image(frameid)
        lidar = self._get_lidar(frameid)
        calib = self._get_calib(frameid)
        target = None
        if self.split == "training":
            target = self._get_label(frameid)

        input_dict = {"image": image, "lidar": lidar, "calib": calib}
        if self.transforms is not None:
            input_dict, target = self.transforms(input_dict, target)
        return input_dict, target

    def _get_image(self, frameid):
        filename = os.path.join(self.image_path, "{:06d}.png".format(frameid))
        image = Image.open(filename)
        return image

    def _get_lidar(self, frameid):
        filename = os.path.join(self.lidar_path, "{:06d}.bin".format(frameid))
        lidar = np.fromfile(filename, dtype=np.float32).reshape(-1, 4)
        return lidar

    def _get_calib(self, frameid):
        filename = os.path.join(self.calib_path, "{:06d}.txt".format(frameid))
        return self.parse_kitti_calib(filename)

    def _get_label(self, frameid):
        filename = os.path.join(self.label_path, "{:06d}.txt".format(frameid))
        return self.parse_kitti_label(filename)

    def parse_kitti_calib(self, filename):
        calib = {}
        with open(filename, "r") as fp:
            lines = [line.strip().split(" ") for line in fp.readlines()]
        calib["P0"] = np.array([float(x) for x in lines[0][1:13]]).reshape(3, 4)
        calib["P1"] = np.array([float(x) for x in lines[1][1:13]]).reshape(3, 4)
        calib["P2"] = np.array([float(x) for x in lines[2][1:13]]).reshape(3, 4)
        calib["P3"] = np.array([float(x) for x in lines[3][1:13]]).reshape(3, 4)
        calib["R0"] = np.array([float(x) for x in lines[4][1:10]]).reshape(3, 3)
        calib["velo_to_cam"] = np.array([float(x) for x in lines[5][1:13]]).reshape(3, 4)
        calib["imu_to_velo"] = np.array([float(x) for x in lines[6][1:13]]).reshape(3, 4)
        return calib

    def parse_kitti_label(self, filename):
        annotations = {
            "name": [],
            "truncated": [],
            "occluded": [],
            "alpha": [],
            "bbox": [],
            "size": [],
            "position": [],
            "yaw": [],
        }
        with open(filename, "r") as fp:
            lines = [line.strip().split(" ") for line in fp.readlines()]
        num_objects = len([x[0] for x in lines if x[0] != "DontCare"])
        annotations["name"] = np.array([x[0] for x in lines])
        annotations["truncated"] = np.array([float(x[1]) for x in lines])
        annotations["occluded"] = np.array([int(x[2]) for x in lines])
        annotations["alpha"] = np.array([float(x[3]) for x in lines])
        annotations["bbox"] = np.array([[float(v) for v in x[4:8]] for x in lines])
        annotations["bbox"] = annotations["bbox"].reshape(-1, 4)
        annotations["size"] = np.array([[float(v) for v in x[8:11]] for x in lines])
        annotations["size"] = annotations["size"].reshape(-1, 3)
        annotations["position"] = np.array([[float(v) for v in x[11:14]] for x in lines])
        annotations["position"] = annotations["position"].reshape(-1, 3)
        annotations["yaw"] = np.array([float(x[14]) for x in lines])
        annotations["yaw"] = annotations["yaw"].reshape(-1)
        return annotations
