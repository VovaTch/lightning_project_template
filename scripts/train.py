import hydra
from omegaconf import DictConfig

import loaders
import models
from utils.learning import LearningParameters, get_trainer


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig) -> None:

    # Get loader
    data_module = getattr(loaders, cfg.data.type).from_cfg(cfg)

    # Get model
    module = getattr(models, cfg.model.module).from_cfg(cfg, weights=cfg.resume)

    # Get trainer
    learning_params = LearningParameters.from_cfg(cfg.model_name, cfg)
    trainer = get_trainer(learning_params)

    # Fit model
    trainer.fit(module, data_module)


if __name__ == "__main__":
    main()
