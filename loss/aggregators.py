from dataclasses import dataclass
from typing import Any, Protocol

import torch

from utils.others import RegisterClass

from .components import LossComponent, COMPONENT_FACTORIES


@dataclass
class LossOutput:
    """
    Loss output object that contains individual components named, and the total loss from the aggregator.

    Fields:
        *   total (Tensor): Total loss from the aggregator
        *   individuals (dict[str, Tensor]): Individual loss component values.
    """

    total: torch.Tensor
    individuals: dict[str, torch.Tensor]


@dataclass
class LossAggregator(Protocol):
    """
    Loss aggregator protocol, uses a math operation on component losses to compute a total loss. For example, weighted sum.
    """

    components: list[LossComponent]

    def __call__(
        self,
        estimation: dict[str, torch.Tensor],
        target: dict[str, torch.Tensor],
    ) -> LossOutput:
        """Call method for loss aggregation

        Args:
            estimation (dict[str, torch.Tensor]): Network estimation dictionary
            target (dict[str, torch.Tensor]): Target dictionary

        Returns:
            LossOutput: LossOutput object representing the total loss and the individual parts
        """
        ...


AGGREGATORS: dict[str, type[LossAggregator]] = {}


@RegisterClass(AGGREGATORS, "weighted_sum")
@dataclass
class WeightedSumAggregator:
    """
    Weighted sum loss component
    """

    components: list[LossComponent]

    def __call__(
        self, estimation: dict[str, torch.Tensor], target: dict[str, torch.Tensor]
    ) -> LossOutput:
        """
        Forward method to compute the weighted sum

        Args:
            estimation (dict[str, torch.Tensor]): Network estimation
            target (dict[str, torch.Tensor]): Ground truth reference

        Returns:
            LossOutput: Loss output object with total loss and individual losses
        """
        loss = LossOutput(torch.tensor(0.0), {})

        for component in self.components:
            ind_loss = component(estimation, target)
            loss.total = loss.total.to(ind_loss.device)
            loss.total += component.weight * ind_loss
            loss.individuals[component.name] = ind_loss

        return loss


def build_loss_aggregator(cfg: dict[str, Any]) -> LossAggregator:
    """
    Builds a loss aggregator object

    Args:
        cfg (dict[str, Any]): Configuration dictionary

    Returns:
        LossAggregator: Loss aggregation object
    """
    aggregator_type = cfg["loss"]["aggregator_type"]
    components = []
    for component_key in cfg["loss"]:
        if component_key == "aggregator_type":
            continue
        component_cfg = cfg["loss"][component_key]
        component_type = component_cfg["type"]
        component = COMPONENT_FACTORIES[component_type](component_key, component_cfg)
        components.append(component)

    return AGGREGATORS[aggregator_type](components)
