import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np

from train import BasicNet

def get_cifar10_data(batch_size):
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # load data
    test_set = torchvision.datasets.CIFAR10(
        root='./data',
        train=False,
        download=False,
        transform=transform
    )

    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size,
                                              shuffle=False, num_workers=2)

    return test_loader


def test_model(model, test_loader):
    test_pred = []

    for data, target in test_loader:
        with torch.no_grad():
            logits = model(data)
            test_pred.extend(logits.argmax(dim=1))

    return np.array(test_pred)


def main():
    model = torch.load('output/cnn_model.pt')
    model.eval()
    test_loader = get_cifar10_data(64)
    test_pred = test_model(model, test_loader)

    np.savetxt("output/test_prediction.csv",
               test_pred,
               delimiter=", ",
               fmt='% s')


if __name__ == '__main__':
    main()
