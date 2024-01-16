from __future__ import annotations
from abc import abstractmethod
from typing import Any, Callable

import lightning as L
from lightning.pytorch.utilities.types import STEP_OUTPUT, OptimizerLRScheduler
import torch
import torch.nn as nn
from loss.aggregators import LossAggregator, LossOutput

from utils.containers import LearningParameters


OptimizerBuilder = Callable[[L.LightningModule], torch.optim.Optimizer]
SchedulerBuilder = Callable[[L.LightningModule], torch.optim.lr_scheduler._LRScheduler]

ACTIVATION_FUNCTIONS: dict[str, nn.Module] = {"relu": nn.ReLU(), "gelu": nn.GELU()}
MODELS: dict[str, nn.Module] = {}
LIGHTNING_MODULES: dict[str, L.LightningModule] = {}


class BaseLightningModule(L.LightningModule):
    """
    Base Pytorch Lightning Module to handle training, validation, testing, logging into Tensorboard, etc.
    The model itself is passed as a Pytorch Module, so this Lightning Module is not limited to a single model.
    """

    def __init__(
        self,
        model: nn.Module,
        learning_params: LearningParameters,
        transforms: nn.Sequential | None = None,
        loss_aggregator: LossAggregator | None = None,
        optimizer_builder: OptimizerBuilder | None = None,
        scheduler_builder: SchedulerBuilder | None = None,
    ) -> None:
        """
        Constructor method

        Args:
            *   model (nn.Module): Base Pytorch model
            *   learning_params (LearningParameters): Learning parameters object containing all parameters required for
                learning.
            *   transforms (nn.Sequential | None, optional): Image transformation sequence, if None, no
                transforms are performed. Defaults to None.
            *   loss_aggregator (LossAggregator | None, optional): Loss object that is composed of multiple components.
                If None, raises an exception when attempting to train. Defaults to None.
            *   optimizer_builder (OptimizerBuilder | None, optional): Optimizer builder function.
                Programmed in this way because it requires a model, the function is called during initialization.
                If None, then AdamW is used. Defaults to None.
            *   scheduler_builder (SchedulerBuilder | None, optional): Scheduler builder function.
                Programmed in this way because it requires a model, the function is called during initialization.
                If None, no scheduler is used. Defaults to None.
        """
        super().__init__()
        self.model = model
        self.learning_params = learning_params
        self.loss_aggregator = loss_aggregator
        self.transforms = transforms

        # Build optimizer and scheduler
        if optimizer_builder is not None:
            self.optimizer = optimizer_builder(self)
        else:
            self.optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=learning_params.learning_rate,
                weight_decay=learning_params.weight_decay,
            )

        self.scheduler = (
            scheduler_builder(self) if scheduler_builder is not None else None
        )

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
        # Pytorch-Lightning specific

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
            TypeError: Must include a scheduler

        Returns:
            dict[str, Any]: Scheduler configuration dictionary
        """
        if self.scheduler is None:
            raise TypeError("Must include a scheduler")
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
            RuntimeError: For training, an optimizer is required (usually shouldn't come to this).
            RuntimeError: For training, must include a loss aggregator.

        Returns:
            STEP_OUTPUT: total loss output
        """
        if self.optimizer is None:
            raise RuntimeError("For training, an optimizer is required.")
        if self.loss_aggregator is None:
            raise RuntimeError("For training, must include a loss aggregator.")
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
        for ind_loss, value in loss.individuals.items():
            self.log(
                f"test_{ind_loss}", value, prog_bar=True, on_step=False, on_epoch=True
            )
        self.log(f"test_total", loss.total, prog_bar=True, on_step=False, on_epoch=True)

    def step(self, batch: dict[str, Any], phase: str) -> torch.Tensor | None:
        """
        Utility method to perform the network step and inference.

        Args:
            batch (dict[str, Any]): Data batch in a form of a dictionary
            phase (str): Phase, used for logging purposes.

        Returns:
            torch.Tensor | None: Either the total loss if there is a loss aggregator, or none if there is no aggregator.
        """
        output = self.forward(batch)
        if self.loss_aggregator is None:
            return
        loss = self.loss_aggregator(output, batch)
        loss_total = self.handle_loss(loss, phase)
        return loss_total

    @abstractmethod
    def handle_loss(self, loss: LossOutput, phase: str) -> torch.Tensor:
        """
        Utility method to implement in a subclass. Used for logging and additional computations if needed.

        Args:
            loss (LossOutput): Loss output object
            phase (str): Phase for logging or other purposes.

        Returns:
            torch.Tensor: Total loss
        """
        ...
