from __future__ import annotations
from abc import abstractmethod
from typing import Any, Callable
from matplotlib.font_manager import weight_dict

import pytorch_lightning as pl
from pytorch_lightning.utilities.types import STEP_OUTPUT, OptimizerLRScheduler
import torch
import torch.nn as nn
from loss.aggregators import LossAggregator, LossOutput

from utils.containers import LearningParameters


OptimizerBuilder = Callable[[pl.LightningModule], torch.optim.Optimizer]
SchedulerBuilder = Callable[[pl.LightningModule], torch.optim.lr_scheduler._LRScheduler]

ACTIVATION_FUNCTIONS: dict[str, nn.Module] = {"relu": nn.ReLU(), "gelu": nn.GELU()}


class BaseLightningModule(pl.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        learning_params: LearningParameters,
        loss_aggregator: LossAggregator | None = None,
        optimizer_builder: OptimizerBuilder | None = None,
        scheduler_builder: SchedulerBuilder | None = None,
    ) -> None:
        super().__init__()
        self.model = model
        self.learning_params = learning_params
        self.loss_aggregator = loss_aggregator

        # Build optimizer and scheduler
        if optimizer_builder is not None:
            self.optimizer = optimizer_builder(self)
        else:
            self.optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=learning_params.learning_rate,
                weight_decay=learning_params.weight_decay,
            )
        if scheduler_builder is not None:
            self.scheduler = scheduler_builder(self)

    def configure_optimizers(self) -> OptimizerLRScheduler:
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
        ...

    def training_step(self, batch: dict[str, Any], batch_idx: int) -> STEP_OUTPUT:
        if self.optimizer is None:
            raise RuntimeError("For training, an optimizer is required.")
        if self.loss_aggregator is None:
            raise RuntimeError("For training, must include a loss aggregator.")
        return self.step(batch)  # type: ignore

    def validation_step(
        self, batch: dict[str, Any], batch_idx: int
    ) -> STEP_OUTPUT | None:
        return self.step(batch)

    def step(self, batch: dict[str, Any]) -> torch.Tensor | None:
        output = self.forward(batch)
        if self.loss_aggregator is None:
            return
        loss = self.loss_aggregator(output, batch)
        loss_total = self.handle_loss(loss)
        return loss_total

    @abstractmethod
    def handle_loss(self, loss: LossOutput) -> torch.Tensor:
        ...
