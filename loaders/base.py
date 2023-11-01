from enum import Enum, auto
from typing import Any, Callable

import pytorch_lightning as pl
from torch.utils.data import Dataset


class Stage(Enum):
    """
    An enum representing different stages in a typical machine learning pipeline.

    - `TRAIN`: Used for the training phase where model parameters are learned.
    - `VALIDATION`: Used for the validation phase to tune hyperparameters and evaluate the model's performance.
    - `TEST`: Used for the testing phase to evaluate the model's performance on unseen data.
    """

    TRAIN = auto()
    VALIDATION = auto()
    TEST = auto()


DataModuleBuilder = Callable[[dict[str, Any]], type[pl.LightningDataModule]]
DATA_MODULES: dict[str, DataModuleBuilder] = {}
