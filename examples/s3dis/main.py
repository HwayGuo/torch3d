import tqdm
import argparse
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import torch3d.transforms as transforms
from torch3d.metrics import Accuracy, Jaccard
from torch3d.models.segmentation import PointNet, JSIS3D
from dataset import S3DIS


def main(args):
    # ToTensor transform converts numpy point clouds into tensors
    transform = transforms.ToTensor()
    dataloaders = {
        "train": data.DataLoader(
            S3DIS(
                args.root,
                train=True,
                test_area=args.test_area,
                transform=transform,
                download=True,  # download dataset if needed
            ),
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=True,
            shuffle=True,
        ),
        "test": data.DataLoader(
            S3DIS(
                args.root,
                train=False,  # now we use the test set
                test_area=args.test_area,
                transform=transform,
                download=False,
            ),
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=True,
            shuffle=False,
        ),
    }
    # Create PointNet model as the backbone
    backbone = PointNet(args.in_channels, args.num_classes)
    model = JSIS3D(backbone, args.num_classes, args.embedding_size).to(args.device)

    optimizer = optim.Adam(model.parameters(), args.lr)
    criteria = loss_fn

    # Here comes the training loop
    for epoch in range(args.epochs):
        train_epoch(args, epoch, model, dataloaders["train"], optimizer, criteria)
        evaluate(args, model, dataloaders["test"])
    print("Done.")


def loss_fn(output, target):
    loss = 0
    loss += F.cross_entropy(output[0], target[..., 0])
    return loss


def train_epoch(args, epoch, model, dataloader, optimizer, criteria):
    desc = "Epoch [{:03d}/{:03d}]".format(epoch, args.epochs)
    pbar = tqdm.tqdm(total=len(dataloader), desc=desc)

    model.train()
    for i, (input, target) in enumerate(dataloader):
        input = input.to(args.device)
        target = target.to(args.device)

        optimizer.zero_grad()
        output = model(input)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()

        pbar.set_postfix(loss=loss.item())
        pbar.update()
    pbar.close()


def evaluate(args, model, dataloader):
    stats = {}
    desc = "Evaluation"
    pbar = tqdm.tqdm(total=len(dataloader), desc=desc)
    # Create metrics to measure the performance, here we use accuracy and IoU
    metrics = [Accuracy(args.num_classes), Jaccard(args.num_classes)]

    model.eval()
    for i, (input, target) in enumerate(dataloader):
        input = input.to(args.device)
        target = target.to(args.device)
        output = model(input)

        # Metrics are updated after every loop
        for metric in metrics:
            metric.update(output[0], target[..., 0])
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
    parser.add_argument("--epochs", default=100, type=int)
    # Use "--device cpu" if your computer does not have CUDA
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--in_channels", default=9, type=int)
    parser.add_argument("--num_classes", default=13, type=int)
    parser.add_argument("--embedding_size", default=32, type=int)
    parser.add_argument("--test_area", default=5, type=int)
    parser.add_argument("--lr", default=0.001, type=float)
    args = parser.parse_args()
    main(args)
