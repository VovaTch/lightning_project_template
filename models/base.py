from __future__ import annotations
from abc import abstractmethod
from typing import Any, Callable, Protocol
from typing_extensions import Self

import lightning as L
from lightning.pytorch.utilities.types import STEP_OUTPUT, OptimizerLRScheduler
from omegaconf import DictConfig
import torch
import torch.nn as nn
from loss.aggregators import LossOutput

from utils.learning import LearningParameters


OptimizerBuilder = Callable[[L.LightningModule], torch.optim.Optimizer]
SchedulerBuilder = Callable[[L.LightningModule], torch.optim.lr_scheduler._LRScheduler]

ACTIVATION_FUNCTIONS: dict[str, nn.Module] = {"relu": nn.ReLU(), "gelu": nn.GELU()}


class LossAggregator(Protocol):
    """
    Protocol class for loss aggregator.

    This class defines the protocol for a loss aggregator, which is responsible for aggregating
    the losses calculated by the model.

    Args:
        pred (dict[str, torch.Tensor]): The predicted values from the model.
        target (dict[str, torch.Tensor]): The target values.

    Returns:
        LossOutput: The aggregated loss output.

    """

    def __call__(
        self, pred: dict[str, torch.Tensor], target: dict[str, torch.Tensor]
    ) -> LossOutput: ...


class BaseLightningModule(L.LightningModule):
    """
    Base Lightning module class, to be inherited by all models. Contains the basic structure
    of a Lightning module, including the optimizer and scheduler configuration.
    """

    def __init__(
        self,
        model: nn.Module,
        learning_params: LearningParameters,
        transforms: nn.Sequential | None = None,
        loss_aggregator: LossAggregator | None = None,
        optimizer_cfg: dict[str, Any] | None = None,
        scheduler_cfg: dict[str, Any] | None = None,
    ) -> None:
        """
        Initializes the BaseModel class.

        Args:
        *   model (nn.Module): The neural network model.
        *   learning_params (LearningParameters): The learning parameters for training.
        *   transforms (nn.Sequential | None, optional): The data transforms to be applied. Defaults to None.
        *   loss_aggregator (LossAggregator | None, optional): The loss aggregator for collecting losses.
            Defaults to None.
        *   optimizer_cfg (dict[str, Any] | None, optional): The configuration for the optimizer.
            Defaults to None.
        *   scheduler_cfg (dict[str, Any] | None, optional): The configuration for the scheduler.
            Defaults to None.
        """
        super().__init__()

        self.model = model
        self.learning_params = learning_params
        self.loss_aggregator = loss_aggregator
        self.transforms = transforms

        self.optimizer = self._build_optimizer(optimizer_cfg)
        self.scheduler = self._build_scheduler(scheduler_cfg)

    def _build_optimizer(
        self, optimizer_cfg: dict[str, Any] | None
    ) -> torch.optim.Optimizer:
        """
        Utility method to build the optimizer.

        Args:
            optimizer_cfg (dict[str, Any] | None): Optimizer configuration dictionary.
                The dictionary should contain the following keys:
                - 'type': The type of optimizer to be used (e.g., 'SGD', 'Adam', etc.).
                - Any additional key-value pairs specific to the chosen optimizer.

        Returns:
            torch.optim.Optimizer: The optimizer object.

        Raises:
            AttributeError: If the specified optimizer type is not supported.
        """
        if optimizer_cfg is not None and optimizer_cfg["type"] != "none":
            filtered_optimizer_cfg = {
                key: value for key, value in optimizer_cfg.items() if key != "type"
            }
            optimizer = getattr(torch.optim, optimizer_cfg["type"])(
                self.parameters(), **filtered_optimizer_cfg
            )
        else:
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=self.learning_params.learning_rate,
                weight_decay=self.learning_params.weight_decay,
                amsgrad=True,
            )
        return optimizer

    def _build_scheduler(
        self, scheduler_cfg: dict[str, Any] | None
    ) -> torch.optim.lr_scheduler._LRScheduler | None:
        """
        Utility method to build the scheduler.

        Args:
            scheduler_cfg (dict[str, Any] | None): Scheduler configuration dictionary.

        Returns:
            torch.optim.lr_scheduler._LRScheduler | None: The built scheduler object,
            or None if scheduler_cfg is None.
        """
        # Build scheduler
        if scheduler_cfg is not None and scheduler_cfg["type"] != "none":
            filtered_schedulers_cfg = {
                key: value for key, value in scheduler_cfg.items() if key != "type"
            }
            scheduler = getattr(torch.optim.lr_scheduler, scheduler_cfg["type"])(
                self.optimizer, **filtered_schedulers_cfg
            )
        else:
            scheduler = None
        return scheduler

    def configure_optimizers(self) -> OptimizerLRScheduler:
        """
        Optimizer configuration Lightning module method. If no scheduler, returns only optimizer.
        If there is a scheduler, returns a settings dictionary and returned to be used during training.

        Returns:
            OptimizerLRScheduler: Method output, used internally.
        """
        if self.scheduler is None:
            return [self.optimizer]

        scheduler_settings = self._configure_scheduler_settings(
            self.learning_params.interval,
            self.learning_params.loss_monitor,
            self.learning_params.frequency,
        )
        return [self.optimizer], [scheduler_settings]  # type: ignore

    def _configure_scheduler_settings(
        self, interval: str, monitor: str, frequency: int
    ) -> dict[str, Any]:
        """
        Utility method to return scheduler configurations to `self.configure_optimizers` method.

        Args:
            interval (str): Intervals to use the scheduler, either 'step' or 'epoch'.
            monitor (str): Loss to monitor and base the scheduler on.
            frequency (int): Frequency to potentially use the scheduler.

        Raises:
            AttributeError: Must include a scheduler

        Returns:
            dict[str, Any]: Scheduler configuration dictionary
        """
        if self.scheduler is None:
            raise AttributeError("Must include a scheduler")
        return {
            "scheduler": self.scheduler,
            "interval": interval,
            "monitor": monitor,
            "frequency": frequency,
        }

    @abstractmethod
    def forward(self, input: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        Forward method, to be implemented in a subclass

        Args:
            input (dict[str, torch.Tensor]): Input dictionary of tensors

        Returns:
            dict[str, torch.Tensor]: Output dictionary of tensors
        """
        ...

    def training_step(self, batch: dict[str, Any], batch_idx: int) -> STEP_OUTPUT:
        """
        Pytorch Lightning standard training step. Uses the loss aggregator to compute the total loss.

        Args:
            batch (dict[str, Any]): Data batch in a form of a dictionary
            batch_idx (int): Data index

        Raises:
            AttributeError: For training, an optimizer is required (usually shouldn't come to this).
            AttributeError: For training, must include a loss aggregator.

        Returns:
            STEP_OUTPUT: total loss output
        """
        if self.optimizer is None:
            raise AttributeError("For training, an optimizer is required.")
        if self.loss_aggregator is None:
            raise AttributeError("For training, must include a loss aggregator.")
        return self.step(batch, "training")  # type: ignore

    def validation_step(
        self, batch: dict[str, Any], batch_idx: int
    ) -> STEP_OUTPUT | None:
        """
        Pytorch lightning validation step. Does not require a loss object this time, but can use it.


        Args:
            batch (dict[str, Any]): Data batch in a form of a dictionary
            batch_idx (int): Data index

        Returns:
            STEP_OUTPUT | None: total loss output if there is an aggregator, none if there isn't.
        """
        return self.step(batch, "validation")

    def test_step(self, batch: dict[str, Any], batch_idx: int) -> STEP_OUTPUT | None:
        """
        Pytorch lightning test step. Uses the loss aggregator to compute and display all losses during the test
        if there is an aggregator.

        Args:
            batch (dict[str, Any]): Data batch in a form of a dictionary
            batch_idx (int): Data index

        Returns:
            STEP_OUTPUT | None: total loss output if there is an aggregator, none if there isn't.
        """
        output = self.forward(batch)
        if self.loss_aggregator is None:
            return
        loss = self.loss_aggregator(output, batch)
        for ind_loss, value in loss.individual.items():
            self.log(
                f"test_{ind_loss}", value, prog_bar=True, on_step=False, on_epoch=True
            )
        self.log(
            "test_total",
            loss.total,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )

    @abstractmethod
    def step(self, batch: dict[str, Any], phase: str) -> torch.Tensor | None:
        """
        Utility method to perform the network step and inference.

        Args:
            batch (dict[str, Any]): Data batch in a form of a dictionary
            phase (str): Phase, used for logging purposes.

        Returns:
            torch.Tensor | None: Either the total loss if there is a loss aggregator, or none if there is no aggregator.
        """
        ...

    @classmethod
    @abstractmethod
    def from_cfg(cls, cfg: DictConfig, weights: str | None = None) -> Self:
        """
        Create an instance of the class from a configuration dictionary.

        Args:
            cfg (DictConfig): The configuration dictionary.
            weights (str | None): Path to the weights file to load. Defaults to None.

        Returns:
            Self: An instance of the class.

        """
        ...
