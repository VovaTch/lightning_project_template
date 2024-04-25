from __future__ import annotations
from typing import Any
from typing_extensions import Self

from omegaconf import DictConfig
import torch
import torch.nn as nn

from models.base import ACTIVATION_FUNCTIONS


class FCN(nn.Module):
    """
    Basic fully-connected network to be used for MNIST classification.
    """

    def __init__(
        self,
        hidden_size: int = 256,
        num_layers: int = 2,
        activation_function: nn.Module = nn.LeakyReLU(),
    ) -> None:
        """
        Initialize the model.

        Args:
            hidden_size (int): The number of units in the hidden layer. Default is 256.
            num_layers (int): The number of layers in the model. Must be at least 2. Default is 2.
            activation_function (nn.Module): The activation function to use in the model. Default is nn.LeakyReLU().
        """
        super().__init__()
        if num_layers < 2:
            raise ValueError(
                f"Number of layers must be at least 2, got {num_layers} number of layers"
            )
        layer_list = (
            [
                nn.Linear(28 * 28, hidden_size),
                activation_function,
            ]
            + [
                nn.Linear(hidden_size, hidden_size),
                activation_function,
            ]
            * (num_layers - 2)
            + [nn.Linear(hidden_size, 10)]
        )
        self.network = nn.Sequential(*layer_list)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        return self.network(x.flatten(start_dim=1))

    @classmethod
    def from_cfg(cls, cfg: DictConfig) -> Self:
        """
        Create an instance of the class using the provided configuration.

        Args:
            cfg (DictConfig): The configuration dictionary.

        Returns:
            Self: An instance of the class.

        """
        return cls(
            hidden_size=cfg.model.params.hidden_size,
            num_layers=cfg.model.params.num_layers,
            activation_function=ACTIVATION_FUNCTIONS[
                cfg.model.params.activation_function
            ],
        )


def fcn(model_name: str, weights_path: str | None = None) -> FCN:
    """
    Create an instance of the FCN model based on the specified model name and optional weights path.

    Args:
        model_name (str): The name of the model configuration to use. Must be one of "small" or "large".
        weights_path (str | None, optional): The path to the weights file to load. Defaults to None.

    Returns:
        FCN: An instance of the FCN model.

    Raises:
        ValueError: If the specified model name is not found in the configurations.

    """
    configurations: dict[str, dict[str, Any]] = {
        "small": {
            "hidden_size": 128,
            "num_layers": 2,
            "activation_function": "relu",
        },
        "large": {
            "hidden_size": 256,
            "num_layers": 3,
            "activation_function": "relu",
        },
    }

    if model_name not in configurations:
        raise ValueError(
            f"Model name {model_name} not found in configurations, available configurations: {list(configurations.keys())}"
        )

    model = FCN(**configurations[model_name])
    if weights_path is not None:
        model.load_state_dict(torch.load(weights_path))
    return model
