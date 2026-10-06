from __future__ import annotations

import lightning as L
import pytest
from hydra import compose, initialize
from hydra.utils import instantiate
from omegaconf import DictConfig

from models.base import BaseLightningModule
from utils.learning import LearningParameters, get_trainer


@pytest.fixture
def cfg() -> DictConfig:
    """
    Composes the default Hydra config with single-process data loading.

    Returns:
    *   DictConfig: Composed configuration.
    """
    with initialize(version_base=None, config_path="../config"):
        return compose("config", overrides=["learning.num_workers=0"])


def test_config_instantiates(cfg: DictConfig) -> None:
    learning_params = instantiate(cfg.learning)
    module = instantiate(cfg.module, _convert_="partial")
    assert isinstance(learning_params, LearningParameters)
    assert isinstance(module, BaseLightningModule)
    assert isinstance(get_trainer(learning_params), L.Trainer)


def test_config_fast_dev_run(cfg: DictConfig) -> None:
    data_module = instantiate(cfg.data)
    module = instantiate(cfg.module, _convert_="partial")
    trainer = L.Trainer(
        fast_dev_run=True,
        accelerator="cpu",
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    trainer.fit(module, data_module)
    trainer.test(module, data_module)
