from __future__ import annotations
from typing import Any
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS

from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
import pytorch_lightning as pl

from utils.containers import LearningParameters, parse_learning_parameters_from_cfg

from .base import Stage


class MnistDataset(Dataset):
    def __init__(self, stage: Stage, data_path: str) -> None:
        super().__init__()
        if stage == Stage.TRAIN:
            self.base_dataset = datasets.MNIST(data_path, train=True, download=True)
        elif stage == Stage.VALIDATION or stage == Stage.TEST:
            self.base_dataset = datasets.MNIST(data_path, train=False, download=True)

    def __getitem__(self, index: int) -> dict[str, Any]:
        data = self.base_dataset.__getitem__(index)
        return {"images": data}

    def __len__(self):
        return len(self.base_dataset)


class SeparatedSetModule(pl.LightningDataModule):
    def __init__(
        self,
        learning_params: LearningParameters,
        train_dataset: Dataset,
        val_dataset: Dataset,
        test_dataset: Dataset | None = None,
    ) -> None:
        super().__init__()
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset if test_dataset is not None else val_dataset
        self.learning_params = learning_params

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(
            self.train_dataset,
            batch_size=self.learning_params.batch_size,
            shuffle=True,
            num_workers=self.learning_params.num_workers,
        )

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            self.val_dataset,
            batch_size=self.learning_params.batch_size,
            shuffle=False,
            num_workers=self.learning_params.num_workers,
        )

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            self.test_dataset,
            batch_size=self.learning_params.batch_size,
            shuffle=False,
            num_workers=self.learning_params.num_workers,
        )


def build_mnist_data_module(cfg: dict[str, Any]) -> SeparatedSetModule:
    learning_params = parse_learning_parameters_from_cfg(cfg)
    dataset_path = cfg["data_path"]
    train_dataset = MnistDataset(Stage.TRAIN, dataset_path)
    val_dataset = MnistDataset(Stage.VALIDATION, dataset_path)
    return SeparatedSetModule(learning_params, train_dataset, val_dataset)
