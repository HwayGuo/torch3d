# Torch3d

[![Build Status](https://img.shields.io/travis/pqhieu/torch3d?style=flat-square)](https://travis-ci.com/pqhieu/torch3d)
[![codecov](https://img.shields.io/codecov/c/github/pqhieu/torch3d?style=flat-square)](https://codecov.io/gh/pqhieu/torch3d)
[![PyPI](https://img.shields.io/pypi/v/torch3d?style=flat-square)](https://pypi.org/project/torch3d)
[![License](https://img.shields.io/github/license/pqhieu/torch3d?style=flat-square)](LICENSE)

Torch3d is a PyTorch library consisting of datasets, model architectures, and
common operations for 3D deep learning. For 3D domain, there is currently no
official support from PyTorch that likes [torchvision](https://github.com/pytorch/vision)
for images. Torch3d aims to fill this gap by streamlining the prototyping
process of deep learning on 3D domain. Currently, Torch3d focuses on deep
learning methods on 3D point sets.


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

> **NOTE**: Some operators require CUDA.


## Getting started

Here are some examples to get you started. These examples assume that you have a basic understanding of PyTorch.
- [Point cloud classification (ModelNet40) using PointNet](examples/modelnet40) (Beginner)

> Take a look at [SotA3d](https://github.com/pqhieu/sota3d) to see how Torch3d is being used in practice.


## Modules

Torch3d composes of the following modules:
- **datasets**: Common 3D datasets for classification, semantic segmentation, and so on.
  + ModelNet40 [[URL](https://modelnet.cs.princeton.edu/)]
  + S3DIS [[URL](http://buildingparser.stanford.edu/dataset.html)]
  + ShapeNetPart [[URL](https://cs.stanford.edu/~ericyi/project_page/part_annotation/)]
  + SceneNN [[URL](http://scenenn.net/)]
- **metrics**: Metrics for on-the-fly training evaluation of different tasks.
  + Accuracy
  + Jaccard (Intersection-over-Union)
- **models**: State-of-the-art models based on their original papers. The following models are currently supported:
  + PointNet from Qi et al. (CVPR 2017) [[Paper](https://arxiv.org/abs/1612.00593)]
  + PoinNet++ from Qi et al. (NeurIPS 2017) [[Paper](https://arxiv.org/abs/1706.02413)]
  + DGCNN from Wang et al. (ToG 2019) [[Paper](https://arxiv.org/abs/1801.07829)]
- **nn**: Low-level operators that can be used to build up complex 3D neural networks.
- **transforms**: Common transformations for dataset preprocessing.
