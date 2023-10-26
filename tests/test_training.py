import pytest
import pytorch_lightning as pl
from loaders.mnist import SeparatedSetModule, build_mnist_data_module
from models.fcn_mnist import LightningFCN

from utils.containers import parse_learning_parameters_from_cfg
from utils.others import load_config
from utils.trainer import initialize_trainer


@pytest.fixture
def trainer() -> pl.Trainer:
    cfg = load_config("tests/test_config.yaml")
    learning_params = parse_learning_parameters_from_cfg(cfg)
    learning_params.epochs = 1
    return initialize_trainer(learning_params)


@pytest.fixture
def data_module() -> SeparatedSetModule:
    cfg = load_config("tests/test_config.yaml")
    return build_mnist_data_module(cfg)


def test_training(
    trainer: pl.Trainer, data_module: SeparatedSetModule, fcn_lightning: LightningFCN
) -> None:
    trainer.fit(fcn_lightning, data_module)
