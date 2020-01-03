import tqdm
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch3d
import torch3d.models as models
import torch3d.datasets as dsets
import torch3d.transforms as transforms
from torch3d.nn.utils import _single
from torch3d.nn import ChamferLoss
import pyvista as pv


class Folding3d(nn.Sequential):
    def __init__(self, in_channels, out_channels, bias=True):
        self.in_channels = in_channels
        self.out_channels = _single(out_channels)
        self.bias = bias
        in_channels = self.in_channels
        modules = []
        for channels in self.out_channels:
            modules.append(nn.Conv1d(in_channels, channels, 1, bias=self.bias))
            modules.append(nn.ReLU(True))
            in_channels = channels
        modules.append(nn.Conv1d(in_channels, 3, 1, bias=self.bias))
        super(Folding3d, self).__init__(*modules)

    def forward(self, x):
        x = super(Folding3d, self).forward(x)
        return x


class FoldingNet(nn.Module):
    def __init__(self, in_channels, meshgrid=[-0.3, 0.3, 45]):
        super(FoldingNet, self).__init__()
        self.in_channels = in_channels
        self.start = meshgrid[0]
        self.end = meshgrid[1]
        self.steps = meshgrid[2]
        self.grid_size = self.steps ** 2
        self.mlp1 = nn.Sequential(
            nn.Conv1d(in_channels, 64, 1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(True),
            nn.Conv1d(64, 64, 1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(True),
        )
        self.mlp2 = nn.Sequential(
            nn.Conv1d(64, 128, 1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(True),
            nn.Conv1d(128, 1024, 1, bias=False),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
        )
        self.maxpool = nn.AdaptiveMaxPool1d(1)
        self.mlp3 = nn.Sequential(
            nn.Linear(1024, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.Linear(512, 512, bias=False),
        )
        self.fold1 = Folding3d(512 + 2, [512, 512], bias=True)
        self.fold2 = Folding3d(512 + 3, [512, 512], bias=True)

    def forward(self, x):
        batch_size = x.shape[0]
        device = x.device
        # Encode point cloud into codeword
        x = self.mlp1(x)
        x = self.mlp2(x)
        x = self.maxpool(x).squeeze(2)
        x = self.mlp3(x)
        # Decode by folding
        g = torch3d.meshgrid2d(self.start, self.end, self.steps, device=device)
        g = g.unsqueeze(0).repeat(batch_size, 1, 1)
        x = x.unsqueeze(2).repeat(1, 1, self.grid_size)
        g = torch.cat([x, g], dim=1)
        g = self.fold1(g)
        x = torch.cat([x, g], dim=1)
        x = self.fold2(x)
        return x


def main(args):
    # ToTensor transform converts numpy point clouds into tensors
    transform = transforms.ToTensor()
    dataloaders = {
        "train": data.DataLoader(
            dsets.ShapeNetPart(
                args.root,
                split="train",
                transform=transform,
                download=True,  # download dataset if needed
            ),
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=True,
            shuffle=True,
        ),
        "test": data.DataLoader(
            dsets.ShapeNetPart(
                args.root,
                split="test",  # now we use the test set
                transform=transform,
                download=False,
            ),
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=True,
            shuffle=False,
        ),
    }
    # Create FoldingNet model
    model = FoldingNet(args.in_channels).to(args.device)
    optimizer = optim.Adam(model.parameters(), args.lr)
    criteria = ChamferLoss()

    # Showing random samples
    points, _ = next(iter(dataloaders["test"]))
    points = points.transpose(2, 1).numpy()
    visualize(points)

    # Here comes the training loop
    for epoch in range(args.epochs):
        train_epoch(args, epoch, model, dataloaders["train"], optimizer, criteria)
    predict = evaluate(args, model, dataloaders["test"])
    visualize(predict[0])


def train_epoch(args, epoch, model, dataloader, optimizer, criteria):
    desc = "Epoch [{:03d}/{:03d}]".format(epoch, args.epochs)
    pbar = tqdm.tqdm(total=len(dataloader), desc=desc)

    model.train()
    for i, (points, _) in enumerate(dataloader):
        points = points.to(args.device)

        optimizer.zero_grad()
        output = model(points)
        loss = criteria(output, points)
        loss.backward()
        optimizer.step()

        pbar.set_postfix(loss=loss.item())
        pbar.update()
    pbar.close()


def evaluate(args, model, dataloader):
    desc = "Evaluation"
    pbar = tqdm.tqdm(total=len(dataloader), desc=desc)

    model.eval()
    with torch.no_grad():
        predict = []
        for i, (points, _) in enumerate(dataloader):
            points = points.to(args.device)
            output = model(points)

            predict.append(output.cpu().transpose(2, 1).numpy())
            pbar.update()
        pbar.close()
    return predict


def visualize(points, predict=None):
    pv.set_plot_theme("document")
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
    plt.link_views()
    plt.camera_position = [
        (5.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
    ]
    plt.show("ShapeNet")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data", type=str)
    # You can increase the training epochs for better performance
    parser.add_argument("--epochs", default=10, type=int)
    # Use "--device cpu" if your computer does not have CUDA
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--in_channels", default=3, type=int)
    parser.add_argument("--lr", default=0.0001, type=float)
    args = parser.parse_args()
    main(args)
