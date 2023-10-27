from enum import Enum, auto
from typing import Any, Callable

import pytorch_lightning as pl
from torch.utils.data import Dataset


class Stage(Enum):
    TRAIN = auto()
    VALIDATION = auto()
    TEST = auto()


DataModuleBuilder = Callable[[dict[str, Any]], type[pl.LightningDataModule]]
DATA_MODULES: dict[str, DataModuleBuilder] = {}
