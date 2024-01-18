from dataclasses import dataclass
from typing import Any, Type
from omegaconf import DictConfig

import torch
import torch.nn as nn
from torchvision import transforms as T

from loss.aggregators import LossOutput, build_loss_aggregator
from utils.containers import parse_learning_parameters_from_cfg
from utils.others import register_builder
from .base import ACTIVATION_FUNCTIONS, BaseLightningModule, MODELS, LIGHTNING_MODULES


@dataclass
class FCNParams:
    """
    Data class representing parameters for a Fully Connected Network (FCN) architecture.

    Args:
        hidden_size (int): The number of units (neurons) in each hidden layer of the FCN.
        num_layers (int): The total number of hidden layers in the FCN.
        activation_function (nn.Module): The activation function to be used in the hidden layers.

    Attributes:
        hidden_size (int): The number of units (neurons) in each hidden layer of the FCN.
        num_layers (int): The total number of hidden layers in the FCN.
        activation_function (nn.Module): The activation function to be used in the hidden layers.

    Note:
        The `FCNParams` data class is designed to encapsulate the architectural parameters of a
        fully connected neural network (FCN). It is typically used to configure and initialize an
        FCN in deep learning applications.
    """

    hidden_size: int
    num_layers: int
    activation_function: nn.Module


def parse_fcn_params_from_cfg(cfg: DictConfig) -> FCNParams:
    """
    Parse the simple Fully Connected Network (FCN) parameters from a configuration dictionary.

    Args:
        cfg (DictConfig): A dictionary containing configuration parameters for the FCN architecture.

    Returns:
        FCNParams: An instance of the `FCNParams` data class containing FCN architecture parameters.

    Note:
        This function takes a configuration dictionary and extracts the architectural
        parameters required to define a Fully Connected Neural Network (FCN). It returns an `FCNParams`
        instance that encapsulates the hidden size, number of layers, and activation function to be used in
        the FCN architecture.
    """
    return FCNParams(
        hidden_size=cfg.model.hidden_size,
        num_layers=cfg.model.num_layers,
        activation_function=ACTIVATION_FUNCTIONS[cfg.model.activation_function],
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


@register_builder(MODELS, "mnist_fcn")
def build_fcn_network(cfg: DictConfig) -> FCN:
    """
    Builds an FCN network for this example from a configuration dictionary

    Args:
        cfg (DictConfig): Configuration dictionary

    Returns:
        FCN: Model instance returned
    """
    fcn_params = parse_fcn_params_from_cfg(cfg)
    return FCN(fcn_params)


class LightningFCN(BaseLightningModule):
    """
    An implementation of the simple FCN network for Mnist purposes.
    """

    def forward(self, input: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        Forward method; assumes 'images' is in the input dictionary, outputs a dictionary with 'pred_logits' key.

        Args:
            input (dict[str, torch.Tensor]): Input dictionary

        Returns:
            dict[str, torch.Tensor]: Output dictionary
        """
        if self.transforms is not None:
            input["images"] = self.transforms(input["images"])
        output_logits = self.model(input["images"])
        return {"pred_logits": output_logits}

    def handle_loss(self, loss: LossOutput, phase: str) -> torch.Tensor:
        """
        Implementation of the loss handling function, just logs all losses into Tensorboard.

        Args:
            loss (LossOutput): Loss output object
            phase (str): Phase for logging purposes

        Returns:
            torch.Tensor: Total loss
        """
        for key, value in loss.individuals.items():
            self.log(phase + "_" + key, value)
        self.log(phase + "_total_loss", loss.total, prog_bar=True)
        return loss.total


@register_builder(LIGHTNING_MODULES, "mnist_fcn")
def build_fcn_lightning_module(
    cfg: DictConfig, weights: str | None = None
) -> LightningFCN:  # TODO: Add optimizers and schedulers
    """
    Builds an FCN lightning module given configuration dictionary and possible weights.

    Args:
        *   cfg (DictConfig): Configuration dictionary
        *   weights (str | None, optional): Weights path to load, if None, then Lightning initializes the weights
            randomly. Defaults to None.

    Returns:
        LightningFCN: Lightning FCN instance.
    """
    fcn_network = build_fcn_network(cfg)
    learning_params = parse_learning_parameters_from_cfg(cfg)
    loss_aggregator = build_loss_aggregator(cfg)
    transforms = T.Compose([T.Normalize((0.1307,), (0.3081,))])
    if weights is None:
        return LightningFCN(
            fcn_network,
            learning_params,
            loss_aggregator=loss_aggregator,
            transforms=transforms,  # type: ignore
        )
    else:
        return LightningFCN.load_from_checkpoint(
            weights,
            model=fcn_network,
            learning_params=learning_params,
            loss_aggregator=loss_aggregator,
            transforms=transforms,
        )  # type: ignore)
