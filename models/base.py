from __future__ import annotations

import os
from abc import abstractmethod
from functools import partial
from logging import Logger
from typing import Any, Callable, Iterator, Protocol, Sequence, TypeVar

import lightning as L
import torch
import torch.nn as nn
from lightning.pytorch.utilities.types import (
    STEP_OUTPUT,
    LRSchedulerConfigType,
    OptimizerLRScheduler,
)
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from loss.aggregators import LossOutput
from utils.learning import LearningParameters
from utils.logger import LOGGER

T = TypeVar("T")
OptimizerFactory = Callable[[Iterator[nn.Parameter]], Optimizer]
SchedulerFactory = Callable[[Optimizer], LRScheduler]


class LossAggregator(Protocol):
    """
    Structural type of a loss aggregator, as used by the Lightning module.
    """

    def __call__(
        self, pred: dict[str, torch.Tensor], target: dict[str, torch.Tensor]
    ) -> LossOutput:
        """
        Args:
        *   pred (dict[str, torch.Tensor]): The predicted values.
        *   target (dict[str, torch.Tensor]): The target values.

        Returns:
        *   LossOutput: Total and individual losses.
        """
        ...


def _as_sequence(items: Sequence[T] | T | None) -> list[T]:
    """
    Normalizes a single item, a sequence of items, or None into a list.

    Args:
    *   items (Sequence[T] | T | None): Input item(s).

    Returns:
    *   list[T]: Items as a list, empty if None.
    """
    if items is None:
        return []
    if isinstance(items, Sequence):
        return list(items)
    return [items]


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
        optimizers: Sequence[OptimizerFactory] | OptimizerFactory | None = None,
        schedulers: Sequence[SchedulerFactory] | SchedulerFactory | None = None,
        logger: Logger = LOGGER,
    ) -> None:
        """
        Initializes the BaseLightningModule class.

        Args:
        *   model (nn.Module): The neural network model.
        *   learning_params (LearningParameters): The learning parameters for training.
        *   transforms (nn.Sequential | None, optional): The data transforms. Defaults to None.
        *   loss_aggregator (LossAggregator | None, optional): The loss aggregator. Defaults to None.
        *   optimizers (Sequence[OptimizerFactory] | OptimizerFactory | None, optional):
            Optimizer factories, called with the parameters. Defaults to None (AdamW).
        *   schedulers (Sequence[SchedulerFactory] | SchedulerFactory | None, optional):
            Scheduler factories, called with the matching optimizer. Defaults to None.
        *   logger (Logger, optional): Python logger. Defaults to LOGGER.
        """
        super().__init__()

        self.model = model
        self.learning_params = learning_params
        self.loss_aggregator = loss_aggregator
        self.transforms = transforms

        if optimizers is None:
            optimizers = partial(
                torch.optim.AdamW,
                lr=learning_params.learning_rate,
                weight_decay=learning_params.weight_decay,
                amsgrad=True,
            )
            logger.info("No optimizer given, defaulting to AdamW.")

        # Factories only; built in `configure_optimizers` after compile/wrapping.
        self._optimizer_factories = _as_sequence(optimizers)
        self._scheduler_factories = _as_sequence(schedulers)
        self._logger = logger

    def _setup_optimizers(
        self, optimizers: Sequence[OptimizerFactory]
    ) -> list[Optimizer]:
        """
        Setups optimizers for the model. For custom logic, override this method.

        Args:
        *   optimizers (Sequence[OptimizerFactory]): Optimizer factories.

        Returns:
        *   list[Optimizer]: Built optimizers.
        """
        return [optimizer(self.parameters()) for optimizer in optimizers]

    def _setup_schedulers(
        self,
        schedulers: Sequence[SchedulerFactory],
        optimizers: Sequence[Optimizer],
    ) -> list[LRScheduler]:
        """
        Setups learning rate schedulers, i-th scheduler wraps i-th optimizer.
        Either no schedulers, or one per optimizer. For custom logic, override this method.

        Args:
        *   schedulers (Sequence[SchedulerFactory]): Scheduler factories.
        *   optimizers (Sequence[Optimizer]): Built optimizers.

        Raises:
        *   ValueError: Scheduler count is neither 0 nor the optimizer count.

        Returns:
        *   list[LRScheduler]: Built schedulers.
        """
        if len(schedulers) not in (0, len(optimizers)):
            raise ValueError(
                f"Got {len(schedulers)} schedulers for {len(optimizers)} optimizers."
            )
        return [
            scheduler(optimizer) for scheduler, optimizer in zip(schedulers, optimizers)
        ]

    def configure_optimizers(self) -> OptimizerLRScheduler:
        """
        Lightning optimizer configuration. Builds every optimizer and pairs it with
        its scheduler (if any), using the scheduler settings from the learning params.

        Returns:
        *   OptimizerLRScheduler: Optimizers, or optimizer + scheduler config dicts.
        """
        optimizers = self._setup_optimizers(self._optimizer_factories)
        schedulers = self._setup_schedulers(self._scheduler_factories, optimizers)
        if not schedulers:
            return optimizers
        return [
            {
                "optimizer": optimizer,
                "lr_scheduler": self._configure_scheduler_settings(scheduler),
            }
            for optimizer, scheduler in zip(optimizers, schedulers)
        ]

    def _configure_scheduler_settings(
        self, scheduler: LRScheduler
    ) -> LRSchedulerConfigType:
        """
        Builds a Lightning scheduler configuration dict from the learning params.

        Args:
        *   scheduler (LRScheduler): Scheduler to configure.

        Returns:
        *   LRSchedulerConfigType: Scheduler configuration dictionary.
        """
        return {
            "scheduler": scheduler,
            "interval": self.learning_params.interval,
            "monitor": self.learning_params.loss_monitor,
            "frequency": self.learning_params.frequency,
        }

    @abstractmethod
    def forward(self, input: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        Forward method, to be implemented in a subclass

        Args:
        *   input (dict[str, torch.Tensor]): Input dictionary of tensors

        Returns:
        *   dict[str, torch.Tensor]: Output dictionary of tensors
        """
        ...

    def training_step(self, batch: dict[str, Any], batch_idx: int) -> STEP_OUTPUT:
        """
        Pytorch Lightning standard training step. Uses the loss aggregator to compute the total loss.

        Args:
        *   batch (dict[str, Any]): Data batch in a form of a dictionary
        *   batch_idx (int): Data index

        Raises:
        *   AttributeError: For training, must include a loss aggregator.

        Returns:
        *   STEP_OUTPUT: total loss output
        """
        if self.loss_aggregator is None:
            raise AttributeError("For training, must include a loss aggregator.")
        return self.step(batch, "training")

    def validation_step(
        self, batch: dict[str, Any], batch_idx: int
    ) -> STEP_OUTPUT | None:
        """
        Pytorch lightning validation step. Does not require a loss object this time, but can use it.


        Args:
        *   batch (dict[str, Any]): Data batch in a form of a dictionary
        *   batch_idx (int): Data index

        Returns:
        *   STEP_OUTPUT | None: total loss output if there is an aggregator, none if there isn't.
        """
        return self.step(batch, "validation")

    def test_step(self, batch: dict[str, Any], batch_idx: int) -> STEP_OUTPUT | None:
        """
        Pytorch lightning test step. Uses the loss aggregator to compute and display all losses during the test
        if there is an aggregator.

        Args:
        *   batch (dict[str, Any]): Data batch in a form of a dictionary
        *   batch_idx (int): Data index

        Returns:
        *   STEP_OUTPUT | None: total loss output if there is an aggregator, none if there isn't.
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
        *   batch (dict[str, Any]): Data batch in a form of a dictionary
        *   phase (str): Phase, used for logging purposes.

        Returns:
        *   torch.Tensor | None: Either the total loss if there is a loss aggregator, or none if there is no aggregator.
        """
        ...


def load_inner_model_state_dict(
    module: BaseLightningModule, checkpoint_path: str
) -> BaseLightningModule:
    """
    Loads a Lightning checkpoint's state dict into an uncompiled module. Strips the
    `_orig_mod.` prefix so checkpoints saved from a `torch.compile`d model also load.
    Call before compiling the module.

    Args:
    *   module (BaseLightningModule): The base lightning module.
    *   checkpoint_path (str): The path to the checkpoint file.

    Raises:
    *   FileNotFoundError: Checkpoint file does not exist.

    Returns:
    *   BaseLightningModule: The base lightning module with the loaded state dictionary.
    """
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found at {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = {
        key.replace("_orig_mod.", ""): value
        for key, value in checkpoint["state_dict"].items()
    }
    module.load_state_dict(state_dict)
    return module
