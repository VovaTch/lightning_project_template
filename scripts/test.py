from __future__ import annotations

import os

import hydra
import torch
from omegaconf import DictConfig

from models.base import BaseLightningModule, load_inner_model_state_dict
from utils.learning import get_trainer


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig) -> None:
    """
    Evaluates the saved checkpoint on the configured test set.

    Args:
    *   cfg (DictConfig): Composed Hydra configuration.
    """

    # Set seed and precision
    torch.manual_seed(1337)
    torch.set_float32_matmul_precision("high")

    # Get weights path
    weights_path = os.path.join(cfg.learning.save_path, cfg.model_name + ".ckpt")

    # Get loader
    data_module = hydra.utils.instantiate(cfg.data)

    # Get lightning module
    module: BaseLightningModule = hydra.utils.instantiate(
        cfg.module, _convert_="partial"
    )
    module = load_inner_model_state_dict(module, weights_path)
    if cfg.use_torch_compile:
        module.model.compile()

    # Get trainer
    learning_params = hydra.utils.instantiate(cfg.learning)
    trainer = get_trainer(learning_params)

    # Fit model
    trainer.test(module, data_module)


if __name__ == "__main__":
    main()
