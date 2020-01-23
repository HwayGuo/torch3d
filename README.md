# Torch3d

[![Build Status](https://img.shields.io/travis/pqhieu/torch3d)](https://travis-ci.com/pqhieu/torch3d)
[![codecov](https://img.shields.io/codecov/c/github/pqhieu/torch3d)](https://codecov.io/gh/pqhieu/torch3d)
[![PyPI](https://img.shields.io/pypi/v/torch3d)](https://pypi.org/project/torch3d)
[![License](https://img.shields.io/github/license/pqhieu/torch3d)](LICENSE)

## Why Torch3d?

Torch3d is a PyTorch library consisting of datasets, model architectures, and common operations for 3D deep learning.
For 3D domain, there is currently no official support from PyTorch that likes [torchvision](https://github.com/pytorch/vision) for images.
Torch3d aims to fill this gap by streamlining the prototyping process of deep learning on 3D domain.


## Installation

Required PyTorch 1.2 or newer. Some other dependencies are:
- torchvision
- h5py

From PyPi:
```bash
$ pip install torch3d
```

From source:
```bash
$ git clone https://github.com/pqhieu/torch3d
$ cd torch3d
$ pip install --editable .
```


## Getting started

Here are some examples to get you started. These examples assume that you have a basic understanding of PyTorch.
- [Point cloud classification (ModelNet40) using PointNet](examples/modelnet40) (Beginner)
- [Point cloud auto-encoder with FoldingNet](examples/foldingnet) (Beginner)


## Modules

Torch3d composes of the following modules:
- **datasets**: Common 3D datasets for classification, semantic segmentation, and so on.
  + ModelNet40 [[URL](https://modelnet.cs.princeton.edu/)] (classification)
  + S3DIS [[URL](http://buildingparser.stanford.edu/dataset.html)] (semantic segmentation)
  + ShapeNet [[URL](https://cs.stanford.edu/~ericyi/project_page/part_annotation/)] (part segmentation)
  + SceneNN [[URL](http://scenenn.net/)] (semantic segmentation)
- **metrics**: Metrics for on-the-fly training evaluation of different tasks.
  + Accuracy (classification, segmentation)
  + IoU (segmentation)
- **models**: State-of-the-art models based on their original papers. The following models are currently supported:
  + PointNet from Qi et al. (CVPR 2017) [[Paper](https://arxiv.org/abs/1612.00593)]
  + PoinNet++ from Qi et al. (NeurIPS 2017) [[Paper](https://arxiv.org/abs/1706.02413)]
  + DGCNN from Wang et al. (ToG 2019) [[Paper](https://arxiv.org/abs/1801.07829)]
  + PointCNN from Li et al. (NeurIPS 2018) [[Paper](https://arxiv.org/abs/1801.07791)]
  + FoldingNet from Yang et al. (CVPR 2018) [[Paper](https://arxiv.org/abs/1712.07262)]
  + PointConv from Wu et al. (CVPR 2019) [[Paper](https://arxiv.org/abs/1811.07246)]
- **nn**: Low-level operators that can be used to build up complex 3D neural networks.
- **transforms**: Common transformations for dataset preprocessing.
