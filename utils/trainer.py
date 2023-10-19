import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    ModelSummary,
    LearningRateMonitor,
)
from pytorch_lightning.loggers import TensorBoardLogger

from .ema import EMA
from .containers import LearningParameters


def initialize_trainer(learning_parameters: LearningParameters) -> pl.Trainer:
    """
    Initializes a Pytorch Lightning training, given a learning parameters object

    Args:
        learning_parameters (LearningParameters): learning parameters object

    Returns:
        pl.Trainer: Pytorch lightning trainer
    """
    # Set device
    num_devices = learning_parameters.num_devices
    accelerator = "cpu" if num_devices == 0 else "gpu"
    device_list = [idx for idx in range(num_devices)]

    # Configure trainer
    ema = EMA(learning_parameters.beta_ema)
    learning_rate_monitor = LearningRateMonitor(logging_interval="step")
    logger = TensorBoardLogger(name=learning_parameters.model_name, save_dir="saved/")
    model_checkpoint_callback = ModelCheckpoint(
        save_last=True,
        save_weights_only=True,
        save_top_k=1,
        monitor=learning_parameters.loss_monitor,
    )

    # AMP
    precision = 16 if learning_parameters.amp else 32

    model_summary = ModelSummary(max_depth=3)
    trainer = pl.Trainer(
        gradient_clip_val=learning_parameters.gradient_clip,
        logger=logger,
        callbacks=[
            model_checkpoint_callback,
            model_summary,
            learning_rate_monitor,
            ema,
        ],
        devices=device_list,
        max_epochs=learning_parameters.epochs,
        log_every_n_steps=1,
        precision=precision,
        accelerator=accelerator,
    )

    return trainer
