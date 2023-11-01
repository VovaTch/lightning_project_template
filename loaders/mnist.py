from __future__ import annotations
from typing import Any
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS

from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
import torchvision.transforms.functional as TF
import pytorch_lightning as pl

from utils.containers import LearningParameters, parse_learning_parameters_from_cfg
from utils.others import register_builder
from .base import Stage, DATA_MODULES


class MnistDataset(Dataset):
    """
    Basic Mnist dataset, loads either the training or the validation sets (for val and test.)
    """

    def __init__(self, stage: Stage, data_path: str, preload: bool = False) -> None:
        """
        Constructor method

        Args:
            stage (Stage): Stage of the training
            data_path (str): Path of the data for the model
            preload (bool, optional): Pre-load the model, here it's unused. Defaults to False.
        """
        super().__init__()
        if stage == Stage.TRAIN:
            self.base_dataset = datasets.MNIST(data_path, train=True, download=True)
        elif stage == Stage.VALIDATION or stage == Stage.TEST:
            self.base_dataset = datasets.MNIST(data_path, train=False, download=True)

    def __getitem__(self, index: int) -> dict[str, Any]:
        """
        Standard Pytorch getitem method, gets a dictionary of tensors as a data.

        Args:
            index (int): Index of data to get

        Returns:
            dict[str, Any]: A data dictionary with 2 entries:
                -   'images' for image data
                -   'class' for ground truth classes
        """
        data = self.base_dataset.__getitem__(index)
        return {"images": TF.to_tensor(data[0]), "class": data[1]}

    def __len__(self) -> int:
        """
        Length of the dataset

        Returns:
            int: Dataset length
        """
        return len(self.base_dataset)


import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset


class SeparatedSetModule(pl.LightningDataModule):
    """
    LightningDataModule subclass for managing separated datasets (train, validation, and test) in a PyTorch Lightning project.

    Args:
        learning_params (LearningParameters): A data class or dictionary containing various learning parameters.
        train_dataset (Dataset): The dataset used for training.
        val_dataset (Dataset): The dataset used for validation.
        test_dataset (Dataset | None, optional): The dataset used for testing. If not provided,
        the validation dataset is used for testing.

    Attributes:
        train_dataset (Dataset): The training dataset.
        val_dataset (Dataset): The validation dataset.
        test_dataset (Dataset): The testing dataset.

    Note:
        The `SeparatedSetModule` class is designed to manage and provide data loaders for the training,
        validation, and testing phases of a deep learning project.
    """

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

    def train_dataloader(self) -> DataLoader:
        """
        Returns a DataLoader for the training dataset.

        Returns:
            DataLoader: A DataLoader for the training dataset.
        """
        return DataLoader(
            self.train_dataset,
            batch_size=self.learning_params.batch_size,
            shuffle=True,
            num_workers=self.learning_params.num_workers,
        )

    def val_dataloader(self) -> DataLoader:
        """
        Returns a DataLoader for the validation dataset.

        Returns:
            DataLoader: A DataLoader for the validation dataset.
        """
        return DataLoader(
            self.val_dataset,
            batch_size=self.learning_params.batch_size,
            shuffle=False,
            num_workers=self.learning_params.num_workers,
        )

    def test_dataloader(self) -> DataLoader:
        """
        Returns a DataLoader for the testing dataset.

        Returns:
            DataLoader: A DataLoader for the testing dataset.
        """
        return DataLoader(
            self.test_dataset,
            batch_size=self.learning_params.batch_size,
            shuffle=False,
            num_workers=self.learning_params.num_workers,
        )


@register_builder(DATA_MODULES, "mnist")
def build_mnist_data_module(cfg: dict[str, Any]) -> SeparatedSetModule:
    """
    Build and return a `SeparatedSetModule` for managing MNIST datasets based on the provided configuration.

    Args:
        cfg (dict): A dictionary containing configuration parameters for building the data module.

    Returns:
        SeparatedSetModule: A `SeparatedSetModule` instance for managing the MNIST datasets during training, validation, and testing.

    Note:
    *   This function is a builder for creating a `SeparatedSetModule` tailored for MNIST dataset management.
        It parses learning parameters from the configuration, sets up train and validation datasets,
        and returns a data module ready for use in a PyTorch Lightning project.
    """
    learning_params = parse_learning_parameters_from_cfg(cfg)
    dataset_path = cfg["data_path"]
    train_dataset = MnistDataset(Stage.TRAIN, dataset_path)
    val_dataset = MnistDataset(Stage.VALIDATION, dataset_path)
    return SeparatedSetModule(learning_params, train_dataset, val_dataset)
