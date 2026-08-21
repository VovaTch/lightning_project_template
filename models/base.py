from __future__ import annotations

import os
import warnings
from abc import abstractmethod
from functools import partial
from logging import Logger
from typing import Any, Protocol, Sequence

import lightning as L
import torch
import torch.nn as nn
from lightning.pytorch.utilities.types import STEP_OUTPUT, OptimizerLRScheduler
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from loss.aggregators import LossOutput
from utils.learning import LearningParameters
from utils.logger import LOGGER


class LossAggregator(Protocol):
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
        optimizers: Sequence[partial[Optimizer]] | partial[Optimizer] | None = None,
        schedulers: Sequence[partial[LRScheduler]] | partial[LRScheduler] | None = None,
        logger: Logger = LOGGER,
    ) -> None:
        """
        Initializes the BaseModel class.

        Args:
        *   model (nn.Module): The neural network model.
        *   learning_params (LearningParameters): The learning parameters for training.
        *   transforms (nn.Sequential | None, optional): The data transforms to be applied. Defaults to None.
        *   loss_aggregator (LossAggregator | None, optional): The loss aggregator for collecting losses.
            Defaults to None.
        *   optimizer (Optimizer | None, optional): The optimizer to be used. Defaults to None, if so, then AdamW is initialized.
        *   scheduler (LRScheduler | None, optional): The learning rate scheduler. Defaults to None.
        """
        super().__init__()

        self.model = model
        self.learning_params = learning_params
        self.loss_aggregator = loss_aggregator
        self.transforms = transforms

        if optimizers is not None and not isinstance(optimizers, Sequence):
            optimizers = [optimizers]
        elif optimizers is None:
            optimizers = [
                partial(
                    torch.optim.AdamW,
                    lr=learning_params.learning_rate,
                    weight_decay=learning_params.weight_decay,
                    amsgrad=True,
                )
            ]

        self._optimizers = self._setup_optimizers(optimizers)

        if schedulers is not None and not isinstance(schedulers, Sequence):
            schedulers = [schedulers]

        self._schedulers = self._setup_schedulers(schedulers)
        self._logger = logger

    def _setup_optimizers(
        self, optimizers: Sequence[partial[Optimizer]]
    ) -> Sequence[Optimizer]:
        """
        Setups optimizers for the model. For custom logic, implement a custom method.

        Args:
            optimizers (Sequence[partial[Optimizer]]): Optimizers to use.

        Returns:
            Sequence[Optimizer]: Sequence of optimizers
        """
        return [optimizer(self.parameters()) for optimizer in optimizers]

    def _setup_schedulers(
        self, schedulers: Sequence[partial[LRScheduler]] | None
    ) -> Sequence[LRScheduler] | None:
        """
        Setups learning rate schedulers for the model. For custom logic, implement a custom method.

        Args:
            schedulers (Sequence[partial[LRScheduler]] | None): Schedulers to use.

        Returns:
            Sequence[LRScheduler] | None: Sequence of schedulers, None if no schedulers defined.
        """
        if schedulers is None:
            return None
        return [
            scheduler(optimizer)
            for scheduler, optimizer in zip(schedulers, self._optimizers)
        ]

    def configure_optimizers(self) -> OptimizerLRScheduler:
        """
        Optimizer configuration Lightning module method. If no scheduler, returns only optimizer.
        If there is a scheduler, returns a settings dictionary and returned to be used during training.

        Returns:
            OptimizerLRScheduler: Method output, used internally.
        """
        self._logger.info(
            "Using default optimizer; to customize, implement a custom 'configure_optimizers'"
        )

        if self._schedulers is None:
            return [self._optimizers[0]]
        else:
            return {
                "optimizer": self._optimizers[0],
                "lr_scheduler": self._schedulers[0],
            }

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
        if self._schedulers is None:
            raise AttributeError("Must include a scheduler")
        return {
            "scheduler": self._schedulers[0],
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
        if not self._optimizers:
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
                f"test_{ind_loss}",
                value,
                prog_bar=True,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
                batch_size=self.learning_params.batch_size,
            )
        self.log(
            "test_total",
            loss.total,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            batch_size=self.learning_params.batch_size,
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


def load_inner_model_state_dict(
    module: BaseLightningModule, checkpoint_path: str
) -> BaseLightningModule:
    """
    Loads the state dictionary of the inner model from a checkpoint file. If the checkpoint file is not found,
    or if an error occurs while loading the checkpoint, the model is returned without loading pre-trained weights.

    Args:
        module (BaseLightningModule): The base lightning module.
        checkpoint_path (str): The path to the checkpoint file.

    Returns:
        BaseLightningModule: The base lightning module with the loaded state dictionary.

    """
    if not os.path.isfile(checkpoint_path):
        warnings.warn(
            f"Checkpoint file not found at {checkpoint_path}, skipping weight loading."
        )
        return module

    try:
        checkpoint = torch.load(checkpoint_path)
        state_dict = checkpoint["state_dict"]
        module.load_state_dict(state_dict)

    except Exception as e:
        warnings.warn(f"Error loading checkpoint: {e}, loading model without weights.")

    finally:
        return module
