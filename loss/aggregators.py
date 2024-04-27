from dataclasses import dataclass
from typing import Iterable, Protocol

import torch


@dataclass
class LossOutput:
    """
    Represents the output of a loss calculation.

    Attributes:
        total (torch.Tensor): The total loss value.
        individual (dict[str, torch.Tensor]): A dictionary containing individual loss values for each component.
    """

    total: torch.Tensor
    individual: dict[str, torch.Tensor]


class LossComponent(Protocol):
    """
    Represents a loss component used for aggregating losses in a model.

    Attributes:
        name (str): The name of the loss component.
        differentiable (bool): Indicates whether the loss component is differentiable.
        weight (float): The weight of the loss component.
    """

    name: str
    differentiable: bool
    weight: float

    def __call__(
        self, pred: dict[str, torch.Tensor], target: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Calculate the loss aggregation.

        Args:
            pred (dict[str, torch.Tensor]): The predicted values.
            target (dict[str, torch.Tensor]): The target values.

        Returns:
            torch.Tensor: The aggregated loss.
        """
        ...


class WeightedSumAggregator:
    """
    Aggregator that computes the weighted sum of multiple loss components.

    Args:
        components (Iterable[LossComponent]): A collection of loss components.

    Returns:
        LossOutput: The aggregated loss output.

    Example:

    ```python
    aggregator = WeightedSumAggregator([component1, component2])
    loss_output = aggregator(pred, target)
    ```
    """

    def __init__(self, components: Iterable[LossComponent]) -> None:
        """
        Initializes the Aggregator object.

        Args:
            components (Iterable[LossComponent]): An iterable of LossComponent objects.
        """
        self.components = components

    def __call__(
        self, pred: dict[str, torch.Tensor], target: dict[str, torch.Tensor]
    ) -> LossOutput:
        """
        Calculates the aggregated loss based on the predictions and targets.

        Args:
            pred (dict[str, torch.Tensor]): A dictionary containing the predicted values.
            target (dict[str, torch.Tensor]): A dictionary containing the target values.

        Returns:
            LossOutput: An instance of the LossOutput class representing the aggregated loss.
        """
        loss = LossOutput(torch.tensor(0.0), {})

        for component in self.components:
            ind_loss = component(pred, target)
            if component.differentiable:
                loss.total = loss.total.to(ind_loss.device)
                loss.total += component.weight * ind_loss
            loss.individual[component.name] = ind_loss

        return loss
