from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from .base import ACTIVATION_FUNCTIONS


@dataclass
class FCNParams:
    hidden_size: int
    num_layers: int
    activation_function: nn.Module


def parse_fcn_params_from_cfg(cfg: dict[str, Any]) -> FCNParams:
    return FCNParams(
        hidden_size=cfg["model_fcn"]["hidden_size"],
        num_layers=cfg["model_fcn"]["num_layers"],
        activation_function=ACTIVATION_FUNCTIONS[
            cfg["model_fcn"]["activation_function"]
        ],
    )


class FCN(nn.Module):
    """
    Basic fully connected network to train MNIST on
    """

    def __init__(self, fcn_params: FCNParams) -> None:
        """
        Constructor method

        Args:
            fcn_params (FCNParams): Network parameter object

        Raises:
            ValueError: Number of layers must be at least 2
        """
        super().__init__()
        if fcn_params.num_layers < 2:
            raise ValueError(
                f"Number of layers must be at least 2, got {fcn_params.num_layers} number of layers"
            )
        layer_list = (
            [
                nn.Linear(28 * 28, fcn_params.hidden_size),
                fcn_params.activation_function,
            ]
            + [
                nn.Linear(fcn_params.hidden_size, fcn_params.hidden_size),
                nn.LayerNorm(fcn_params.hidden_size),
                fcn_params.activation_function,
            ]
            * (fcn_params.num_layers - 2)
            + [nn.Linear(fcn_params.hidden_size, 10)]
        )
        self.network = nn.Sequential(*layer_list)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Network forward method

        Args:
            x (torch.Tensor): BS x 28 x 28 tensor image

        Returns:
            torch.Tensor: BS x 10 logit tensor
        """
        return self.network(x.flatten(start_dim=1))


def build_fcn_network(cfg: dict[str, Any]) -> FCN:
    fcn_params = parse_fcn_params_from_cfg(cfg)
    return FCN(fcn_params)
