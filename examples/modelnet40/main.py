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

    # Here comes the training loop
    for epoch in range(args.epochs):
        train_epoch(args, epoch, model, dataloaders["train"], optimizer, criteria)
    evaluate(args, model, dataloaders["test"])
    print("Done.")


def train_epoch(args, epoch, model, dataloader, optimizer, criteria):
    desc = "Epoch [{:03d}/{:03d}]".format(epoch, args.epochs)
    pbar = tqdm.tqdm(total=len(dataloader), desc=desc)

    model.train()
    for i, (input, target) in enumerate(dataloader):
        input = input.to(args.device)
        target = target.to(args.device)

        optimizer.zero_grad()
        output = model(input)
        loss = criteria(output, target)
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
        for i, (input, target) in enumerate(dataloader):
            input = input.to(args.device)
            target = target.to(args.device)
            output = model(input)

            # Metrics are updated after every loop
            for metric in metrics:
                metric.update(output, target)
                stats[metric.name] = metric.score()

            pbar.set_postfix(**stats)
            pbar.update()
        pbar.close()

    for m in metrics:
        print("→ {}: {:.3f} / {:.3f}".format(m.name, m.score(), m.mean()))


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
