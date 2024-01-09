import pathlib

import hydra
import numpy as np
import onnxruntime
import torch
import torchvision
import torchvision.transforms as transforms
from omegaconf import DictConfig, OmegaConf


def get_cifar10_data(batch_size, path_lvl):
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    # load data
    test_set = torchvision.datasets.CIFAR10(
        root=f"{path_lvl}data",
        train=False,
        download=False,
        transform=transform,
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


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    OmegaConf.to_yaml(cfg)

    if str(pathlib.Path().absolute())[-14:] == "mlops-cifar-10":
        path_lvl = "./"
    else:
        path_lvl = "../"

    # get data
    test_loader = get_cifar10_data(
        batch_size=cfg.data_loader.test_batch_size, path_lvl=path_lvl
    )

    # get model
    onnx_session = onnxruntime.InferenceSession(
        f"{path_lvl}output/cnn_classifier.onnx",
        providers=["CPUExecutionProvider"],
    )

    # make prediction and save it
    test_pred = test_model(onnx_session, test_loader)

    np.savetxt(
        f"{path_lvl}output/test_prediction.csv",
        test_pred,
        delimiter=", ",
        fmt="% s",
    )


if __name__ == "__main__":
    main()
