import argparse

import hydra
from omegaconf import DictConfig

from loaders import DATA_MODULES
from utils.containers import parse_learning_parameters_from_cfg
from utils.trainer import initialize_trainer
from models import LIGHTNING_MODULES


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Training script for the MNIST template"
    )
    parser.add_argument(
        "-d",
        "--num_devices",
        type=int,
        default=1,
        help="Number of CUDA devices. If 0, use CPU.",
    )
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="config/config.yaml",
        help="Configuration file path",
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="mnist_fcn",
        help="Type of model used for training.",
    )
    parser.add_argument(
        "-dm",
        "--data_module",
        type=str,
        default="mnist",
        help="Type of data module used for training",
    )
    parser.add_argument(
        "-r",
        "--resume",
        type=str,
        default=None,
        help="Checkpoint path to load the model from",
    )
    return parser.parse_args()


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig) -> None:
    args = get_args()

    cfg.learning.num_devices = args.num_devices
    learning_params = parse_learning_parameters_from_cfg(cfg)
    trainer = initialize_trainer(learning_params)
    model = LIGHTNING_MODULES[args.model](cfg, args.resume)
    data_module = DATA_MODULES[args.data_module](cfg)
    trainer.test(model, data_module)


if __name__ == "__main__":
    main()
