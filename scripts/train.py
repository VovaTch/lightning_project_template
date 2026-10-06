from __future__ import annotations

import hydra
import torch
from omegaconf import DictConfig

from models.base import load_inner_model_state_dict
from utils.learning import get_trainer


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig) -> None:
    """
    Trains the configured module on the configured data.

    Args:
    *   cfg (DictConfig): Composed Hydra configuration.
    """

    # Set precision
    torch.set_float32_matmul_precision("high")

    # Get loader
    data_module = hydra.utils.instantiate(cfg.data)

    # Get lightning module
    module = hydra.utils.instantiate(cfg.module, _convert_="partial")
    if cfg.resume is not None:
        module = load_inner_model_state_dict(module, cfg.resume)
    if cfg.use_torch_compile:
        module.model.compile()

    # Get trainer
    learning_params = hydra.utils.instantiate(cfg.learning)
    trainer = get_trainer(learning_params)

    # Fit model
    trainer.fit(module, data_module)


if __name__ == "__main__":
    main()
