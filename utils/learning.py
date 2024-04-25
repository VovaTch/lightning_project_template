from dataclasses import dataclass
import warnings
from typing_extensions import Self

import torch
from omegaconf import DictConfig
import lightning as L
from lightning.pytorch.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    EarlyStopping,
    ModelSummary,
)
from lightning.pytorch.loggers import Logger, TensorBoardLogger

from utils.ema import EMA


@dataclass
class LearningParameters:
    """
    Learning parameters dataclass to contain every parameter required for training,
    excluding the optimizer and the scheduler, which are handled separately.
    """

    project_name: str
    learning_rate: float
    weight_decay: float
    batch_size: int
    epochs: int
    beta_ema: float
    gradient_clip: float
    save_path: str
    amp: bool
    val_split: float
    test_split: float
    devices: str | int | list[int]
    num_workers: int
    loss_monitor: str
    trigger_loss: float
    interval: str
    frequency: int

    @classmethod
    def from_cfg(cls, exp_name: str, cfg: DictConfig) -> Self:
        """
        Create an instance of the LearningParameters class from a configuration dictionary.

        Args:
            cfg (DictConfig): The configuration dictionary containing the parameters.

        Returns:
            Self: An instance of the LearningParameters class.
        """
        return cls(
            project_name=exp_name,
            learning_rate=cfg.learning.learning_rate,
            weight_decay=cfg.learning.weight_decay,
            batch_size=cfg.learning.batch_size,
            epochs=cfg.learning.epochs,
            beta_ema=cfg.learning.beta_ema,
            gradient_clip=cfg.learning.gradient_clip,
            save_path=cfg.learning.save_path,
            amp=cfg.learning.amp,
            val_split=cfg.learning.val_split,
            test_split=cfg.learning.test_split,
            devices=cfg.learning.devices,
            num_workers=cfg.learning.num_workers,
            loss_monitor=cfg.learning.scheduler.loss_monitor,
            trigger_loss=cfg.learning.trigger_loss,
            interval=cfg.learning.scheduler.interval,
            frequency=cfg.learning.scheduler.frequency,
        )


def get_trainer(learning_parameters: LearningParameters) -> L.Trainer:
    """
    Initializes a Pytorch Lightning training, given a learning parameters object

    Args:
        learning_parameters (LearningParameters): learning parameters object

    Returns:
        pl.Trainer: Pytorch lightning trainer
    """
    # Set device
    if not torch.cuda.is_available():
        warnings.warn("CUDA is not available, using CPU")
        devices = "auto"
        accelerator = "cpu"
    else:
        devices = learning_parameters.devices
        accelerator = (
            "cpu" if learning_parameters.devices == "cpu" or devices == "cpu" else "gpu"
        )

    save_folder = learning_parameters.save_path

    # Configure trainer
    ema = EMA(learning_parameters.beta_ema)
    learning_rate_monitor = LearningRateMonitor(logging_interval="step")
    tensorboard_logger = TensorBoardLogger(
        save_dir=save_folder, name=learning_parameters.project_name
    )
    loggers: list[Logger] = [tensorboard_logger]

    model_checkpoint_callback = ModelCheckpoint(
        dirpath=save_folder,
        filename=learning_parameters.project_name,
        save_weights_only=True,
        save_top_k=1,
        monitor=learning_parameters.loss_monitor,
        enable_version_counter=False,
    )
    early_stopping = EarlyStopping(
        monitor=learning_parameters.loss_monitor,
        stopping_threshold=learning_parameters.trigger_loss,
        patience=int(
            learning_parameters.epochs
        ),  # Early stopping is here is for only stopping once the training reached a threshold loss.
    )

    # AMP
    precision = 16 if learning_parameters.amp else 32

    model_summary = ModelSummary(max_depth=2)
    trainer = L.Trainer(
        gradient_clip_val=learning_parameters.gradient_clip,
        logger=loggers,
        callbacks=[
            early_stopping,
            model_checkpoint_callback,
            model_summary,
            learning_rate_monitor,
            ema,
        ],
        devices=devices,
        max_epochs=learning_parameters.epochs,
        log_every_n_steps=1,
        precision=precision,
        accelerator=accelerator,
    )

    return trainer
