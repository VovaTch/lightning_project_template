import torch

from models.fcn_mnist import FCN


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
