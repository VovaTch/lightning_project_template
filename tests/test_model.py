import numpy as np
import torch

from models.fcn_mnist import FCN, LightningFCN


def test_fcn_forward_cpu(fcn: FCN) -> None:
    """
    Test the forward pass of the FCN model on CPU.

    Args:
        fcn (FCN): The FCN model to test.
    """
    fcn = fcn
    input = torch.rand((5, 28, 28))
    output = fcn(input)
    assert list(output.size()) == [5, 10]


def test_fcn_forward_cuda(fcn: FCN) -> None:
    """
    Test the forward pass of the FCN model on CUDA.

    Args:
        fcn (FCN): The FCN model to test.
    """
    fcn = fcn.to("cuda")
    input = torch.rand((5, 28, 28)).to("cuda")
    output = fcn(input)
    assert list(output.size()) == [5, 10]


def test_fcn_lightning_forward(fcn_lightning: LightningFCN) -> None:
    """
    Test the forward method of the LightningFCN model.

    Args:
        fcn_lightning (LightningFCN): The LightningFCN model to test.
    """
    fcn_lightning = fcn_lightning.to("cuda")
    input = {"images": torch.rand((5, 28, 28)).to("cuda")}
    output = fcn_lightning.forward(input)
    assert list(output["pred_logits"].size()) == [5, 10]
