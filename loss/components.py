from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Callable, Protocol

import torch
import torch.nn as nn
from torchvision.ops import sigmoid_focal_loss

from utils.others import register_builder


LOSS_MODULES: dict[str, nn.Module] = {
    "mse": nn.MSELoss(),
    "l1": nn.L1Loss(),
    "huber": nn.SmoothL1Loss(),
    "bce": nn.BCEWithLogitsLoss(),
    "ce": nn.CrossEntropyLoss(),
    "focal": sigmoid_focal_loss,  # type: ignore
}


@dataclass
class LossComponent(Protocol):
    """
    Loss component object protocol

    Fields:
        name (str): loss name
        weight (float): loss relative weight for computation (e.g. weighted sum)
        base_loss (nn.Module): base loss module specific for the loss
    """

    name: str
    weight: float
    base_loss: nn.Module
    differentiable: bool

    def __call__(
        self, estimation: dict[str, torch.Tensor], target: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Call method for outputting the loss

        Args:
            estimation (dict[str, torch.Tensor]): Network estimation
            target (dict[str, torch.Tensor]): Ground truth reference

        Returns:
            torch.Tensor: loss
        """
        ...


LossComponentFactory = Callable[[str, dict[str, Any]], LossComponent]

COMPONENT_FACTORIES: dict[str, LossComponentFactory] = {}


# >>>>>>>>>>>>>> ACTUAL COMPONENT IMPLEMENTATION
@dataclass
class BasicClassificationLoss:
    """
    Basic classification loss for classification purposes
    """

    name: str
    weight: float
    base_loss: nn.Module
    differentiable: bool = True

    def __call__(
        self, estimation: dict[str, torch.Tensor], target: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Call method for outputting the loss

        Args:
            estimation (dict[str, torch.Tensor]): Network estimation
            target (dict[str, torch.Tensor]): Ground truth reference

        Returns:
            torch.Tensor: loss
        """
        return self.base_loss(estimation["pred_logits"], target["class"])


@register_builder(COMPONENT_FACTORIES, "basic_cls")
def build_classification_loss(
    name: str, loss_cfg: dict[str, Any]
) -> BasicClassificationLoss:
    """
    Builds basic classification loss with cross-entropy-loss as default.

    Args:
        name (str): Loss name
        loss_cfg (dict[str, Any]): Loss configuration

    Returns:
        BasicClassificationLoss: Basic classification loss object
    """
    loss_module = LOSS_MODULES[loss_cfg.get("base_loss", "ce")]
    return BasicClassificationLoss(name, loss_cfg.get("weight", 1.0), loss_module)


@dataclass
class PercentCorrect:
    """
    Basic metric to count the ratio of the correct number of classifications
    """

    name: str
    weight: float
    base_loss: nn.Module | None = None
    differentiable: bool = False

    def __call__(
        self, estimation: dict[str, torch.Tensor], target: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Call method for outputting the loss

        Args:
            estimation (dict[str, torch.Tensor]): Network estimation
            target (dict[str, torch.Tensor]): Ground truth reference

        Returns:
            torch.Tensor: loss
        """
        pred_logits_argmax = torch.argmax(estimation["pred_logits"], dim=1)
        correct = torch.sum(pred_logits_argmax == target["class"])
        return correct / torch.numel(pred_logits_argmax)


@register_builder(COMPONENT_FACTORIES, "percent_correct")
def build_percent_correct_metric(name: str, loss_cfg: dict[str, Any]) -> PercentCorrect:
    """
    Builds the percent correct metric; this metric does not account into the grad operation

    Args:
        name (str): Loss name
        loss_cfg (dict[str, Any]): Loss configuration

    Returns:
        PercentCorrect: Percent correct loss object
    """
    return PercentCorrect(name, 1.0, None)


# >>>>>>>>>>>>>>>>>>>>>>>>>>> END OF ACTUAL COMPONENT IMPLEMENTATION
