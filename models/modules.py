from typing import Any
from omegaconf import DictConfig
from typing_extensions import Self

import torch
import torch.nn as nn

from loss.aggregators import LossOutput
import loss.aggregators as loss_aggregators
from models.models import FCN
import models.models as models
from utils.learning import LearningParameters

from .base import BaseLightningModule, LossAggregator


class MnistClassifierModule(BaseLightningModule):
    """
    Lightning module for the MNIST classifier, extends the BaseLightningModule class.
    """

    def __init__(
        self,
        model: FCN,
        learning_params: LearningParameters,
        transforms: nn.Sequential | None = None,
        loss_aggregator: LossAggregator | None = None,
        optimizer_cfg: dict[str, Any] | None = None,
        scheduler_cfg: dict[str, Any] | None = None,
    ) -> None:
        """
        Initializes the Module class.

        Args:
            model (FCN): The model to be used.
            learning_params (LearningParameters): The learning parameters.
            transforms (nn.Sequential | None, optional): The data transforms. Defaults to None.
            loss_aggregator (LossAggregator | None, optional): The loss aggregator. Defaults to None.
            optimizer_cfg (dict[str, Any] | None, optional): The optimizer configuration. Defaults to None.
            scheduler_cfg (dict[str, Any] | None, optional): The scheduler configuration. Defaults to None.
        """
        super().__init__(
            model,
            learning_params,
            transforms,
            loss_aggregator,
            optimizer_cfg,
            scheduler_cfg,
        )

    def forward(self, x: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        Forward pass of the MNIST classifier.

        Args:
            x (dict[str, torch.Tensor]): Input data dictionary containing "images" tensor.

        Returns:
            dict[str, torch.Tensor]: Output dictionary containing "logits" tensor.
        """
        outputs = self.model(x["images"])
        return {"logits": outputs}

    def step(self, batch: dict[str, Any], phase: str) -> torch.Tensor | None:
        """
        Performs a single training/validation step.

        Args:
            batch (dict[str, Any]): Input batch data.
            phase (str): The phase of the training (e.g., "train", "val").

        Returns:
            torch.Tensor | None: The total loss if available, otherwise None.
        """
        outputs = self(batch)
        if self.loss_aggregator is not None:
            loss = self.loss_aggregator(outputs, batch)
            loss_total = self.log_loss(loss, phase)
            return loss_total

    def log_loss(self, loss: LossOutput, phase: str) -> torch.Tensor:
        """
        Handles the loss logging (to Tensorboard).

        Args:
            loss (LossOutput): The loss output object containing individual losses.
            phase (str): The phase of the training (e.g., "train", "val").

        Returns:
            torch.Tensor: The total loss.
        """
        for name in loss.individual:
            log_name = f"{phase} {name.replace('_', ' ')}"
            self.log(log_name, loss.individual[name])
        self.log(f"{phase} total loss", loss.total, prog_bar=True)
        return loss.total

    @classmethod
    def from_cfg(cls, cfg: DictConfig, weights: str | None = None) -> Self:
        """
        Creates an instance of the MNIST classifier module from a configuration.

        Args:
            cfg (DictConfig): The configuration object.
            weights (str | None, optional): Path to the pre-trained weights. Defaults to None.

        Returns:
            Self: An instance of the MNIST classifier module.
        """
        model = getattr(models, cfg.model.type).from_cfg(cfg)
        loss_aggregator = (
            getattr(loss_aggregators, cfg.loss.aggregator.type).from_cfg(cfg)
            if cfg.loss.aggregator.type is not None
            else None
        )
        learning_params = LearningParameters.from_cfg(cfg.model_name, cfg)
        scheduler_cfg = cfg.learning.scheduler if "scheduler" in cfg.learning else None
        optimizer_cfg = cfg.learning.optimizer if "optimizer" in cfg.learning else None

        kwargs = {
            "model": model,
            "learning_params": learning_params,
            "loss_aggregator": loss_aggregator,
            "optimizer_cfg": optimizer_cfg,
            "scheduler_cfg": scheduler_cfg,
        }

        if weights is not None:
            return cls.load_from_checkpoint(weights, **kwargs)
        else:
            return cls(**kwargs)
