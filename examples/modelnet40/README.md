# Point cloud classification on ModelNet40

> **Difficulty**: Beginner

Torch3d makes the prototyping process of 3D deep learning a breeze. Here is a
quick tutorial of how to implement a point cloud classifier on the `ModelNet40`
dataset.


## Loading ModelNet40

First we need to import `torch3d.datasets`:

```python
import torch
import torch3d.datasets as dsets
```

The output of the ModelNet40 dataset are `numpy` point clouds, which are
ndarrays of shape `(N, 3)`. We need to convert them into Tensors by using the
`ToTensor` transform:

```python
import torch3d.transforms as transforms

transform = transforms.ToTensor()
```

Here we define two dataloaders, one for the training set, and one for the test set.

```python
dataloaders = {
    "train": data.DataLoader(
        dsets.ModelNet40(
            "data",  # path to the dataset
            train=True,
            transform=transform,
            download=True,  # download the dataset if needed
        ),
        batch_size=64,
        num_workers=4,
        pin_memory=True,
        shuffle=True,
    ),
    "test": data.DataLoader(
        dsets.ModelNet40(
            args.root,
            train=False,  # now we use the test set
            transform=transform,
            download=False,
        ),
        batch_size=64,
        num_workers=4,
        pin_memory=True,
        shuffle=False,
    ),
}
```

Now let's show some samples for inspection. Here we will use
[PyVista](https://docs.pyvista.org/), an excellent Pythonic binding for VTK,
for 3D visualization.

![Dataset](assets/dataset.png?raw=true)

Looking good!
