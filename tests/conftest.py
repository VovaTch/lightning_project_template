import hydra
from omegaconf import DictConfig
import pytest

from models.fcn_mnist import (
    FCN,
    LightningFCN,
    build_fcn_lightning_module,
    build_fcn_network,
)


@pytest.fixture
def cfg() -> DictConfig:
    """
    Fixture that initializes and returns the configuration object.

    Returns:
        DictConfig: The configuration object.
    """
    with hydra.initialize(version_base=None, config_path="./config"):
        # config is relative to a module
        cfg = hydra.compose(config_name="config")
    return cfg


@pytest.fixture
def fcn(cfg: DictConfig) -> FCN:
    """
    Fixture function that returns an instance of the FCN class based on the provided configuration.

    Args:
        cfg (DictConfig): The configuration for building the FCN network.

    Returns:
        FCN: An instance of the FCN class.
    """
    return build_fcn_network(cfg)


@pytest.fixture
def fcn_lightning(cfg: DictConfig) -> LightningFCN:
    """
    Fixture function that returns an instance of LightningFCN
    based on the provided configuration.

    Args:
        cfg (DictConfig): The configuration for building the LightningFCN.

    Returns:
        LightningFCN: An instance of the LightningFCN module.
    """
    return build_fcn_lightning_module(cfg)
