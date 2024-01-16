from omegaconf import DictConfig
import pytest
import lightning as L
from loaders.mnist import SeparatedSetModule, build_mnist_data_module
from models.fcn_mnist import LightningFCN

from utils.containers import parse_learning_parameters_from_cfg
from utils.trainer import initialize_trainer
import hydra


@pytest.fixture
def trainer(cfg: DictConfig) -> L.Trainer:
    """
    Initialize and return a Lightning Trainer object for training.

    Args:
        cfg (DictConfig): The configuration dictionary.

    Returns:
        L.Trainer: The initialized Lightning Trainer object.
    """
    learning_params = parse_learning_parameters_from_cfg(cfg)
    learning_params.epochs = 1
    return initialize_trainer(learning_params)


@pytest.fixture
def data_module(cfg: DictConfig) -> SeparatedSetModule:
    """
    Fixture function that returns a SeparatedSetModule object
    initialized with the provided configuration.

    Args:
        cfg (DictConfig): The configuration for building the data module.

    Returns:
        SeparatedSetModule: The initialized SeparatedSetModule object.
    """
    return build_mnist_data_module(cfg)


def test_training(
    trainer: L.Trainer, data_module: SeparatedSetModule, fcn_lightning: LightningFCN
) -> None:
    """
    Test the training process by fitting the LightningFCN model using the given trainer and data module.

    Args:
        trainer (L.Trainer): The Lightning Trainer object.
        data_module (SeparatedSetModule): The data module containing the training data.
        fcn_lightning (LightningFCN): The LightningFCN model.

    Returns:
        None
    """
    trainer.fit(fcn_lightning, data_module)
