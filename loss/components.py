from dataclasses import dataclass
from typing import Any, Callable
from typing_extensions import Self

from omegaconf import DictConfig
import torch
import torch.nn as nn
from torchvision.ops import sigmoid_focal_loss
import utils.transform_func as transform_func


LOSS_FUNCTIONS = {
    "mse": nn.MSELoss(),
    "l1": nn.L1Loss(),
    "huber": nn.SmoothL1Loss(),
    "bce": nn.BCEWithLogitsLoss(),
    "ce": nn.CrossEntropyLoss(),
    "focal": sigmoid_focal_loss,  # type: ignore
}


@dataclass
class BasicClassificationLoss:
    """
    Basic classification loss, most commonly cross entropy.

    Args:
        name (str): The name of the loss.
        weight (float): The weight of the loss.
        base_loss (nn.Module): The base loss function.
        differentiable (bool, optional): Whether the loss is differentiable. Defaults to True.

    Returns:
        torch.Tensor: The computed loss value.
    """

    name: str
    weight: float
    base_loss: nn.Module
    differentiable: bool = True

    def __call__(
        self, pred: dict[str, torch.Tensor], target: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute the loss.

        Args:
            pred (dict[str, torch.Tensor]): The predicted values.
            target (dict[str, torch.Tensor]): The target values.

        Returns:
            torch.Tensor: The computed loss value.
        """
        return self.base_loss(pred["logits"], target["class"])

    @classmethod
    def from_cfg(cls, name: str, cfg: DictConfig) -> Self:
        """
        Create an instance of BasicClassificationLoss from a configuration.

        Args:
            name (str): The name of the loss.
            cfg (DictConfig): The configuration.

        Returns:
            BasicClassificationLoss: An instance of BasicClassificationLoss.
        """
        return cls(
            name=name,
            weight=cfg.get("weight", 1.0),
            base_loss=LOSS_FUNCTIONS[cfg.get("base_loss", "ce")],
        )


@dataclass
class ReconstructionLoss:
    """
    Reconstruction loss of slices or images most commonly.

    Args:
    *   name (str): The name of the loss.
    *   weight (float): The weight of the loss.
    *   base_loss (nn.Module): The base loss function.
    *   rec_key (str): The key for accessing the reconstruction values in the prediction and target dictionaries.
    *   transform_func (Callable[[torch.Tensor], torch.Tensor], optional):
        The transformation function to apply to the reconstruction values. Defaults to lambda x: x.
    *   differentiable (bool, optional): Whether the loss is differentiable. Defaults to True.

    Returns:
        torch.Tensor: The computed loss value.
    """

    name: str
    weight: float
    base_loss: nn.Module
    rec_key: str
    transform_func: Callable[[torch.Tensor], torch.Tensor] = lambda x: x
    differentiable: bool = True

    def __call__(
        self, pred: dict[str, torch.Tensor], target: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute the loss.

        Args:
            pred (dict[str, torch.Tensor]): The predicted values.
            target (dict[str, torch.Tensor]): The target values.

        Returns:
            torch.Tensor: The computed loss value.
        """
        return self.base_loss(
            self.transform_func(pred[self.rec_key]),
            self.transform_func(target[self.rec_key]),
        )

    @classmethod
    def from_cfg(cls, name: str, cfg: DictConfig) -> Self:
        """
        Create an instance of ReconstructionLoss from a configuration.

        Args:
            name (str): The name of the loss.
            cfg (DictConfig): The configuration.

        Returns:
            ReconstructionLoss: An instance of ReconstructionLoss.
        """
        return cls(
            name=name,
            weight=cfg.get("weight", 1.0),
            base_loss=LOSS_FUNCTIONS[cfg.get("base_loss", "mse")],
            rec_key=cfg.rec_key,
            transform_func=getattr(transform_func, cfg.transform_func),
        )


@dataclass
class PercentCorrect:
    """
    Percent correct metric for classification tasks.

    Args:
        name (str): The name of the metric.
        weight (float): The weight of the metric.
        differentiable (bool, optional): Whether the metric is differentiable. Defaults to False.

    Returns:
        torch.Tensor: The computed metric value.
    """

    name: str
    weight: float
    differentiable: bool = False

    def __call__(
        self, pred: dict[str, torch.Tensor], target: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute the metric.

        Args:
            pred (dict[str, torch.Tensor]): The predicted values.
            target (dict[str, torch.Tensor]): The target values.

        Returns:
            torch.Tensor: The computed metric value.
        """
        pred_logits_argmax = torch.argmax(pred["logits"], dim=1)
        correct = torch.sum(pred_logits_argmax == target["class"])
        return correct / torch.numel(pred_logits_argmax)

    @classmethod
    def from_cfg(cls, name: str, cfg: DictConfig) -> Self:
        """
        Create an instance of PercentCorrect from a configuration.

        Args:
            name (str): The name of the metric.
            cfg (DictConfig): The configuration.

        Returns:
            PercentCorrect: An instance of PercentCorrect.
        """
        return cls(
            name=name,
            weight=cfg.get("weight", 1.0),
        )
