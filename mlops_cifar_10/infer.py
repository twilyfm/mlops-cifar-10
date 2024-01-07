import numpy as np
import onnxruntime
import torch
import torchvision
import torchvision.transforms as transforms


def get_cifar10_data(batch_size):
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    # load data
    test_set = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=False, transform=transform
    )

    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=batch_size, shuffle=False, num_workers=2
    )

    return test_loader


def test_model(onnx_session, test_loader):
    test_pred = []

    for data, target in test_loader:
        ort_inputs = {onnx_session.get_inputs()[0].name: to_numpy(data)}
        ort_outs = onnx_session.run(None, ort_inputs)
        test_pred.extend(torch.Tensor(ort_outs[0]).argmax(dim=1))

    return np.array(test_pred)


def to_numpy(tensor):
    return (
        tensor.detach().cpu().numpy()
        if tensor.requires_grad
        else tensor.cpu().numpy()
    )


def main():
    onnx_session = onnxruntime.InferenceSession(
        "output/cnn_classifier.onnx", providers=["CPUExecutionProvider"]
    )
    test_loader = get_cifar10_data(1)

    test_pred = test_model(onnx_session, test_loader)

    np.savetxt(
        "output/test_prediction.csv", test_pred, delimiter=", ", fmt="% s"
    )


if __name__ == "__main__":
    main()
