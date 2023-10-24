from typing import Any
import torch

from models.fcn_mnist import FCN, LightningFCN


def test_fcn_forward_cpu(fcn: FCN) -> None:
    fcn = fcn
    input = torch.rand((5, 28, 28))
    output = fcn(input)
    assert list(output.size()) == [5, 10]


def test_fcn_forward_cuda(fcn: FCN) -> None:
    fcn = fcn.to("cuda")
    input = torch.rand((5, 28, 28)).to("cuda")
    output = fcn(input)
    assert list(output.size()) == [5, 10]


def test_fcn_lightning_forward(fcn_lightning: LightningFCN) -> None:
    fcn_lightning = fcn_lightning.to("cuda")
    input = {"images": torch.rand((5, 28, 28)).to("cuda")}
    output = fcn_lightning.forward(input)
