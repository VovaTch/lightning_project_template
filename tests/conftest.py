from typing import Any

import pytest

from models.fcn_mnist import (
    FCN,
    LightningFCN,
    build_fcn_lightning_module,
    build_fcn_network,
)
from utils.others import load_config


@pytest.fixture
def cfg() -> dict[str, Any]:
    cfg_path = "tests/test_config.yaml"
    return load_config(cfg_path)


@pytest.fixture
def fcn(cfg: dict[str, Any]) -> FCN:
    return build_fcn_network(cfg)


@pytest.fixture
def fcn_lightning(cfg: dict[str, Any]) -> LightningFCN:
    return build_fcn_lightning_module(cfg)
