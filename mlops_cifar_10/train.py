import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from dvc.api import DVCFileSystem
from sklearn.model_selection import train_test_split


def get_cifar10_data(batch_size):
    torch.manual_seed(0)
    np.random.seed(0)

    DVCFileSystem().get("data", "data", recursive=True)

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    # load data
    trainvalset = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=False, transform=transform
    )

    # split data for train and validation
    train_idx, valid_idx = train_test_split(
        np.arange(len(trainvalset)),
        test_size=0.3,
        shuffle=True,
        random_state=0,
    )

    trainset = torch.utils.data.Subset(trainvalset, train_idx)
    valset = torch.utils.data.Subset(trainvalset, valid_idx)

    # create dataloader
    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=2
    )
    val_loader = torch.utils.data.DataLoader(
        valset, batch_size=batch_size, shuffle=False, num_workers=2
    )

    return train_loader, val_loader


train_loader, val_loader = get_cifar10_data(batch_size=64)


# CNN architecture
class BasicNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=12,
                kernel_size=5,
                stride=3,
                dilation=3,
                padding=1,
            ),
            nn.BatchNorm2d(num_features=12),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=12, out_channels=32, kernel_size=3, padding=1
            ),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=32, out_channels=32, kernel_size=3, padding=1
            ),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=32,
                out_channels=32,
                kernel_size=3,
                stride=3,
                dilation=3,
                padding=1,
            ),
            nn.BatchNorm2d(num_features=32),
        )

        self.classifier = nn.Sequential(
            nn.Linear(in_features=128, out_features=50),
            nn.ReLU(),
            nn.Linear(in_features=50, out_features=10),
        )

    def forward(self, x):
        out_net = self.net(x)
        out_vec = out_net.reshape(out_net.shape[0], -1)
        out = self.classifier(out_vec)

        return out


def validation_epoch(model, loss_func, val_loader):
    loss_log = []
    acc_log = []
    model.eval()

    for data, target in val_loader:
        with torch.no_grad():
            logits = model(data)
            loss = loss_func(logits, target)

        loss_log.append(loss.item())
        acc = torch.sum(logits.argmax(dim=1) == target) / len(target)
        acc_log.append(acc.item())

    return np.mean(loss_log), np.mean(acc_log)


def train_epoch(model, optimizer, loss_func, train_loader):
    loss_log = []
    acc_log = []
    model.train()

    for data, target in train_loader:
        optimizer.zero_grad()
        logits = model(data)

        loss = loss_func(logits, target)
        loss.backward()
        optimizer.step()

        acc = torch.sum(logits.argmax(dim=1) == target) / len(target)
        acc_log.append(acc.item())
        loss_log.append(loss.item())

    return loss_log, acc_log


def train(
    model, optimizer, n_epochs, train_loader, val_loader, scheduler=None
):
    loss_func = nn.CrossEntropyLoss()
    train_loss_log, train_acc_log, val_loss_log, val_acc_log = [], [], [], []

    for epoch in range(n_epochs):
        train_loss, train_acc = train_epoch(
            model, optimizer, loss_func, train_loader
        )
        val_loss, val_acc = validation_epoch(model, loss_func, val_loader)

        train_loss_log.extend(train_loss)
        train_acc_log.extend(train_acc)

        val_loss_log.append(val_loss)
        val_acc_log.append(val_acc)

        print(f"Epoch {epoch}")
        print(
            f"train loss: {np.mean(train_loss)}",
            f"train acc: {np.mean(train_acc)}",
        )
        print(f" val loss: {val_loss}, val acc: {val_acc}\n")

        if scheduler is not None:
            scheduler.step()

    return train_loss_log, train_acc_log, val_loss_log, val_acc_log


def main():
    net = BasicNet()
    optimizer = optim.Adam(net.parameters(), lr=3e-3)

    train_loss_log, train_acc_log, val_loss_log, val_acc_log = train(
        net, optimizer, 1, train_loader, val_loader
    )
    torch_input = torch.randn(1, 3, 32, 32)
    torch.onnx.export(
        net,
        torch_input,
        "output/cnn_classifier.onnx",
        export_params=True,
        do_constant_folding=True,
    )

    # torch.save(net, 'output/cnn_model.pt')


if __name__ == "__main__":
    main()
