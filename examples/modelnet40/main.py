import tqdm
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch3d.models as models
import torch3d.datasets as dsets
import torch3d.transforms as transforms
from torch3d.metrics import Accuracy
import pyvista as pv


def main(args):
    # ToTensor transform converts numpy point clouds into tensors
    transform = transforms.ToTensor()
    dataloaders = {
        "train": data.DataLoader(
            dsets.ModelNet40(
                args.root,
                train=True,
                transform=transform,
                download=True,  # download dataset if needed
            ),
            batch_size=args.batch_size,
            num_workers=args.num_workers,
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
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=True,
            shuffle=False,
        ),
    }
    # Create PointNet model and its optimizer
    model = models.PointNet(args.in_channels, args.num_classes).to(args.device)
    optimizer = optim.Adam(model.parameters(), args.lr)
    criteria = nn.CrossEntropyLoss()

    # Showing random samples
    points, labels = next(iter(dataloaders["test"]))
    points = points.transpose(2, 1).numpy()
    visualize(points, labels)

    # Here comes the training loop
    for epoch in range(args.epochs):
        train_epoch(args, epoch, model, dataloaders["train"], optimizer, criteria)

    predict = evaluate(args, model, dataloaders["test"])
    visualize(points, labels, predict)
    print("Done.")


def train_epoch(args, epoch, model, dataloader, optimizer, criteria):
    desc = "Epoch [{:03d}/{:03d}]".format(epoch, args.epochs)
    pbar = tqdm.tqdm(total=len(dataloader), desc=desc)

    model.train()
    for i, (points, labels) in enumerate(dataloader):
        points = points.to(args.device)
        labels = labels.to(args.device)

        optimizer.zero_grad()
        output = model(points)
        loss = criteria(output, labels)
        loss.backward()
        optimizer.step()

        pbar.update()
    pbar.close()


def evaluate(args, model, dataloader):
    stats = {}
    desc = "Evaluation"
    pbar = tqdm.tqdm(total=len(dataloader), desc=desc)
    # Create metric to measure the performance, here we use accuracy
    metrics = [Accuracy(args.num_classes)]

    model.eval()
    with torch.no_grad():
        predict = None
        for i, (points, labels) in enumerate(dataloader):
            points = points.to(args.device)
            labels = labels.to(args.device)
            output = model(points)

            # Metrics are updated after every loop
            for metric in metrics:
                metric.update(output, labels)
                stats[metric.name] = metric.score()

            if i == 0:
                predict = torch.argmax(output, 1).cpu().numpy()

            pbar.set_postfix(**stats)
            pbar.update()
        pbar.close()

    for m in metrics:
        print("→ {}: {:.3f}".format(m.name, m.score()))
    return predict


def visualize(points, labels, predict=None):
    pv.set_plot_theme("document")
    categories = dsets.ModelNet40.categories
    plt = pv.Plotter(shape=(4, 4), window_size=(800, 800))
    for row in range(4):
        for col in range(4):
            i = row * 4 + col
            plt.subplot(row, col)
            plt.add_mesh(
                points[i],
                render_points_as_spheres=True,
                point_size=8.0,
                ambient=0.1,
                diffuse=0.8,
                specular=0.5,
                specular_power=100.0,
            )
            plt.add_text(
                categories[labels[i]],
                font_size=13,
                position="upper_edge",
                font="arial",
                shadow=False,
            )
            if predict is not None:
                if predict is not None:
                    color = "red" if predict[i] != labels[i] else "green"
                plt.add_text(
                    dsets.ModelNet40.categories[predict[i]],
                    font_size=13,
                    color=color,
                    position="lower_edge",
                    font="arial",
                    shadow=False,
                )
    plt.link_views()
    plt.camera_position = [
        (0.0, 0.0, -5.0),
        (0.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
    ]
    plt.show("ModelNet40")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data", type=str)
    # You can increase the training epochs for better performance
    parser.add_argument("--epochs", default=10, type=int)
    # Use "--device cpu" if your computer does not have CUDA
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--in_channels", default=3, type=int)
    parser.add_argument("--num_classes", default=40, type=int)
    parser.add_argument("--lr", default=0.001, type=float)
    args = parser.parse_args()
    main(args)
