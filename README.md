Torch3d
=======
[![PyPI](https://img.shields.io/pypi/v/torch3d)](https://pypi.org/project/torch3d)
[![Downloads](https://pepy.tech/badge/torch3d)](https://pepy.tech/project/torch3d)

Torch3d is a PyTorch library consisting of datasets, model architectures, and common operations for 3D deep learning.
For 3D domain, there is currently no official support from PyTorch that likes [torchvision](https://github.com/pytorch/vision) for images.
Torch3d aims to fill this gap by streamlining the prototyping process of deep learning on 3D domain.
Currently, Torch3d focuses on deep learning methods on 3D point clouds.


Installation
------------
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

> **Note**: Some of the operations require CUDA.


Tutorials
---------

Torch3d includes some tutorials to get you started.
These tutorials assume that you have a basic understanding of PyTorch.
- [Point cloud classification (ModelNet40) using PointNet](examples/modelnet40) (Beginner)


Modules
-------
- **datasets**: Provide common 3D datasets for classification, semantic segmentation, and so on.
  + ModelNet40 [[URL](https://modelnet.cs.princeton.edu/)]
  + S3DIS [[URL](http://buildingparser.stanford.edu/dataset.html)]
  + ShapeNetPart [[URL](https://cs.stanford.edu/~ericyi/project_page/part_annotation/)]
  + SceneNN [[URL](http://scenenn.net/)]
- **metrics**: Implement different metrics for on-the-fly training evaluation of different tasks.
  + Binary accuracy, accuracy
  + Jaccard (Intersection-over-Union)
- **models**: Re-implement state-of-the-art models based on their original papers. The following models are supported:
  + PointNet from Qi et al. (CVPR 2017) [[Paper](https://arxiv.org/abs/1612.00593)]
  + PoinNet++ from Qi et al. (NeurIPS 2017) [[Paper](https://arxiv.org/abs/1706.02413)]
  + PointCNN from Li et al. (NeurIPS 2018) [[Paper](https://arxiv.org/abs/1801.07791)]
- **nn**: Low-level operations that can be used to build up complex 3D neural networks.
- **transforms**: Common transformations for dataset preprocessing.


Contributions
-------------
All contributions and/or suggestions are welcomed.
