import argparse

from loaders import DATA_MODULES
from utils.containers import parse_learning_parameters_from_cfg
from utils.others import load_config
from utils.trainer import initialize_trainer
from models import LIGHTNING_MODULES


def main(args):
    cfg = load_config(args.config)
    cfg["learn"]["num_devices"] = args.num_devices
    learning_params = parse_learning_parameters_from_cfg(cfg)
    trainer = initialize_trainer(learning_params)
    model = LIGHTNING_MODULES[args.model](cfg)
    data_module = DATA_MODULES[args.data_module](cfg)
    trainer.fit(model, data_module)


if __name__ == "__main__":
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
    args = parser.parse_args()
    main(args)
