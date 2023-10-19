from dataclasses import dataclass
from typing import Any, Callable, Protocol

import torch
import torch.nn as nn
from torchvision.ops import sigmoid_focal_loss


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


# >>>>>>>>>>>>>> ACTUAL COMPONENT IMPLEMENTATION
@dataclass
class BasicClassificationLoss:
    """
    Basic classification loss for classification purposes
    """

    name: str
    weight: float
    base_loss: nn.Module

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
        return self.base_loss(estimation["pred_logits"], target["target_logits"])


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


# >>>>>>>>>>>>>>>>>>>>>>>>>>> END OF ACTUAL COMPONENT IMPLEMENTATION


COMPONENT_FACTORIES: dict[str, LossComponentFactory] = {
    "basic_cls": build_classification_loss
}
