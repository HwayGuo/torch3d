import os
import re
import h5py
import numpy as np
import torch.utils.data as data
from PIL import Image


class KITTIDetection(data.Dataset):
    """The KITTI Detection dataset.

    Args:
      root (string): Root directory of the KITTI dataset.
      split (string, optional): The dataset subset which can be either
        ``"train"``, ``"val"``, or ``"test"``. Default: ``"train"``.
      transforms (callable, optional): A function/transform that takes
        input sample and its target as entry and return a transformed
        version. Default: ``None``.
      rectified (bool, optional): If True, return the LiDAR point cloud
        in the camera rectified coordinate system. Default: ``True``.
      remove_dontcare (bool, optional): If True, remove DontCare label
        from the ground truth annotations. Default: ``True``.
    """

    def __init__(
        self,
        root,
        split="train",
        transforms=None,
        rectified=True,
        remove_dontcare=True,
    ):
        super(KITTIDetection, self).__init__()
        self.root = root
        self.split = split
        if split in ["train", "val"]:
            self.splitdir = "training"
        else:
            self.splitdir = "testing"
        self.transforms = transforms
        self.rectified = rectified
        self.remove_dontcare = remove_dontcare
        self.image_path = os.path.join(root, self.splitdir, "image_2")
        self.lidar_path = os.path.join(root, self.splitdir, "velodyne")
        self.label_path = os.path.join(root, self.splitdir, "label_2")
        self.calib_path = os.path.join(root, self.splitdir, "calib")

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
        if self.rectified:
            calib = self._get_calib(frameid)
            lidar = self.rectify_lidar(lidar, calib)

        inputs = {"image": image, "points": lidar}
        target = None
        if self.split != "test":
            target = self._get_label(frameid)
        if self.transforms is not None:
            inputs, target = self.transforms(inputs, target)
        return inputs, target

    def _get_image(self, frameid):
        basename = "{:06d}.png".format(frameid)
        filename = os.path.join(self.image_path, basename)
        image = Image.open(filename)
        image = np.asarray(image, dtype=np.float32)
        return image

    def _get_lidar(self, frameid):
        basename = "{:06d}.bin".format(frameid)
        filename = os.path.join(self.lidar_path, basename)
        lidar = np.fromfile(filename, dtype=np.float32).reshape(-1, 4)
        return lidar

    def _get_calib(self, frameid):
        basename = "{:06d}.txt".format(frameid)
        filename = os.path.join(self.calib_path, basename)
        calib = self.read_calib(filename)
        return calib

    def _get_label(self, frameid):
        basename = "{:06d}.txt".format(frameid)
        filename = os.path.join(self.label_path, basename)
        annotations = self.read_label_annotations(filename, self.remove_dontcare)
        annotations = self.add_difficulty_to_annotations(annotations)
        return annotations

    @staticmethod
    def rectify_lidar(lidar, calib):
        n = lidar.shape[0]
        velo_to_rect = np.dot(calib["velo_to_cam"].T, calib["R0"].T)
        xyzw = np.c_[lidar[:, 0:3], np.ones(n)]
        xyz = np.dot(xyzw, velo_to_rect)
        intensity = lidar[:, 3]
        return np.c_[xyz, intensity]

    @staticmethod
    def add_difficulty_to_annotations(annotations):
        min_height = [40, 25, 25]
        max_occlusion = [0, 1, 2]
        max_truncation = [0.15, 0.3, 0.5]
        bbox = annotations["bbox"]
        height = bbox[:, 3] - bbox[:, 1]
        occlusion = annotations["occluded"]
        truncation = annotations["truncated"]

        num_annotations = len(annotations["class"])
        is_easy = np.ones((num_annotations,), dtype=np.bool)
        is_moderate = np.ones((num_annotations,), dtype=np.bool)
        is_hard = np.ones((num_annotations,), dtype=np.bool)
        for i, (h, o, t) in enumerate(zip(height, occlusion, truncation)):
            if o > max_occlusion[0] or h <= min_height[0] or t > max_truncation[0]:
                is_easy[i] = False
            if o > max_occlusion[1] or h <= min_height[1] or t > max_truncation[1]:
                is_moderate[i] = False
            if o > max_occlusion[2] or h <= min_height[2] or t > max_truncation[2]:
                is_hard[i] = False
        is_hard = np.logical_xor(is_hard, is_moderate)
        is_moderate = np.logical_xor(is_moderate, is_easy)

        difficulty = []
        for i in range(num_annotations):
            if is_easy[i]:
                difficulty.append(0)
            elif is_moderate[i]:
                difficulty.append(1)
            elif is_hard[i]:
                difficulty.append(2)
            else:
                difficulty.append(-1)
        annotations["difficulty"] = np.array(difficulty, dtype=np.int32)
        return annotations

    @staticmethod
    def read_label_annotations(filename, remove_dontcare=True):
        annotations = {
            "class": [],
            "truncated": [],
            "occluded": [],
            "alpha": [],
            "bbox": [],
            "size": [],
            "center": [],
            "yaw": [],
        }
        with open(filename, "r") as fp:
            lines = [line.strip().split(" ") for line in fp.readlines()]
        if remove_dontcare:
            lines = [line for line in lines if line[0] != "DontCare"]
        annotations["class"] = np.array([x[0] for x in lines])
        annotations["truncated"] = np.array([float(x[1]) for x in lines])
        annotations["occluded"] = np.array([int(x[2]) for x in lines])
        annotations["alpha"] = np.array([float(x[3]) for x in lines])
        annotations["bbox"] = np.array([[float(v) for v in x[4:8]] for x in lines])
        annotations["size"] = np.array([[float(v) for v in x[8:11]] for x in lines])
        annotations["center"] = np.array([[float(v) for v in x[11:14]] for x in lines])
        annotations["yaw"] = np.array([float(x[14]) for x in lines])
        annotations["bbox"] = annotations["bbox"].reshape(-1, 4)
        annotations["size"] = annotations["size"].reshape(-1, 3)
        annotations["center"] = annotations["center"].reshape(-1, 3)
        annotations["yaw"] = annotations["yaw"].reshape(-1)
        return annotations

    @staticmethod
    def read_calib(filename):
        calib = {}
        with open(filename, "r") as fp:
            lines = [line.strip().split(" ") for line in fp.readlines()]
        calib["P0"] = np.array([float(x) for x in lines[0][1:13]])
        calib["P1"] = np.array([float(x) for x in lines[1][1:13]])
        calib["P2"] = np.array([float(x) for x in lines[2][1:13]])
        calib["P3"] = np.array([float(x) for x in lines[3][1:13]])
        calib["R0"] = np.array([float(x) for x in lines[4][1:10]])
        calib["velo_to_cam"] = np.array([float(x) for x in lines[5][1:13]])
        calib["imu_to_velo"] = np.array([float(x) for x in lines[6][1:13]])
        calib["P0"] = calib["P0"].reshape(3, 4)
        calib["P1"] = calib["P1"].reshape(3, 4)
        calib["P2"] = calib["P2"].reshape(3, 4)
        calib["P3"] = calib["P3"].reshape(3, 4)
        calib["R0"] = calib["R0"].reshape(3, 3)
        calib["velo_to_cam"] = calib["velo_to_cam"].reshape(3, 4)
        calib["imu_to_velo"] = calib["imu_to_velo"].reshape(3, 4)
        return calib

    @staticmethod
    def collate_fn(batch):
        return [list(x) for x in zip(*batch)]
