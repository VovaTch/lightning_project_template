from __future__ import annotations

import pytest
import torch

from models.models import FCN


@pytest.fixture
def fcn() -> FCN:
    return FCN()


def test_fcn_forward_pass_cpu(fcn: FCN) -> None:
    input_tensor = torch.randn(32, 1, 28, 28)
    output = fcn.forward(input_tensor)
    assert output.shape == (32, 10)


def test_fcn_forward_pass_gpu(fcn: FCN) -> None:
    input_tensor = torch.randn(32, 1, 28, 28).cuda()
    fcn = fcn.to("cuda")
    output = fcn.forward(input_tensor)
    assert output.shape == (32, 10)


def test_fcn_initialization(fcn: FCN) -> None:
    assert isinstance(fcn, FCN)
    assert len(fcn.network) == 3


def test_fcn_num_layers() -> None:
    try:
        _ = FCN(num_layers=1)
    except ValueError as e:
        assert str(e) == "Number of layers must be at least 2, got 1 number of layers"


def test_fcn_layers_not_shared() -> None:
    model = FCN(num_layers=4)
    linear_ids = [id(m) for m in model.network if isinstance(m, torch.nn.Linear)]
    act_ids = [id(m) for m in model.network if isinstance(m, torch.nn.LeakyReLU)]
    assert len(set(linear_ids)) == 4
    assert len(set(act_ids)) == 3


def test_fcn_factory() -> None:
    from models.models import fcn as fcn_factory

    output = fcn_factory("large")(torch.randn(2, 1, 28, 28))
    assert output.shape == (2, 10)
