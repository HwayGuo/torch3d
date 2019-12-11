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

We define two dataloaders, one for the training set, and one for the test set.

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

Now take a look at some test samples for inspection. Here we will use
[PyVista](https://docs.pyvista.org/), an excellent Pythonic binding for VTK,
for 3D visualization.

![Dataset](assets/dataset.png?raw=true)

Looking good!


## Training a point cloud classifier

We define the network, its optimizer, and the loss function.

```python
import torch.nn as nn
import torch3d.models as models

in_channels = 3  # number of input channels, here we only have XYZ
num_classes = 40
lr = 0.001  # learning rate

model = models.PointNet(in_channels, num_classes)
optimizer = optim.Adam(model.parameters(), lr)
criteria = nn.CrossEntropyLoss()
```

Now we only need to loop over the dataset, feed the inputs to our network, and
optimize.

```python
for epoch in range(10):  # train for 10 epochs
    model.train()
    for i, (points, labels) in enumerate(dataloaders["train"]):
        points = points.to(args.device)
        labels = labels.to(args.device)

        optimizer.zero_grad()
        output = model(points)
        loss = criteria(output, labels)
        loss.backward()
        optimizer.step()
```


## Evaluation

After training, we would love to measure the performance of our network on the
test set. The `torch3d.metrics` package is created for such task:

```python
import torch3d.metrics as metrics

metric = metrics.Accuracy(num_classes)
model.eval()
with torch.no_grad():  # disable autograd to speed up
    for i, (points, labels) in enumerate(dataloaders["test"]):
        points = points.to(args.device)
        labels = labels.to(args.device)
        output = model(points)
        # Metric needs to be updated after every prediction
        metric.update(output, labels)
print("→ accuracy: {:.3f}".format(m.score()))  # print the final score
```

Here we show the test samples again, and their corresponding predictions at the
bottom. We also color-code our results — green for correct and red for
incorrect. Voilà!

![Prediction](assets/predict.png?raw=true)

> **Exercise**: Try training with different networks and compare their performance.
