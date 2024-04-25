import os
import hydra
from omegaconf import DictConfig

import loaders
import models
from utils.learning import LearningParameters, get_trainer


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig) -> None:

    # Get weights path
    weights_path = os.path.join(cfg.learning.save_path, cfg.model_name + ".ckpt")

    # Get loader
    data_module = getattr(loaders, cfg.data.type).from_cfg(cfg)

    # Get model
    module = (
        getattr(models, cfg.model.module).from_cfg(cfg, weights=weights_path).eval()
    )

    # Get trainer
    learning_params = LearningParameters.from_cfg(cfg.model_name, cfg)
    trainer = get_trainer(learning_params)

    # Fit model
    trainer.test(module, data_module)


if __name__ == "__main__":
    main()
