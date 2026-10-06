from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import Dataset

from loaders import SeparatedSetModule
from loss.aggregators import WeightedSumAggregator
from loss.components import BasicClassificationLoss
from models import FCN, MnistClassifierModule
from utils.ema import EMAOptimizer
from utils.learning import LearningParameters, get_trainer


class RandomMnist(Dataset):
    """
    Random MNIST-shaped samples.

    Args:
    *   length (int): Number of samples.
    """

    def __init__(self, length: int) -> None:
        self.length = length

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        """
        Args:
        *   index (int): Sample index (unused).

        Returns:
        *   dict[str, torch.Tensor]: 'images' [1, 28, 28] and 'class' [] tensors.
        """
        return {"images": torch.rand(1, 28, 28), "class": torch.randint(0, 10, ())}

    def __len__(self) -> int:
        return self.length


def test_checkpoint_holds_ema_weights(tmp_path: Path) -> None:
    learning_params = LearningParameters(
        "ema_test",
        batch_size=4,
        epochs=1,
        beta_ema=0.5,
        save_path=str(tmp_path),
        devices=1,
        num_workers=0,
    )
    aggregator = WeightedSumAggregator(
        [BasicClassificationLoss("ce", 1.0, torch.nn.CrossEntropyLoss())]
    )
    module = MnistClassifierModule(FCN(), learning_params, loss_aggregator=aggregator)
    data = SeparatedSetModule(learning_params, RandomMnist(16), RandomMnist(8))

    trainer = get_trainer(learning_params)
    trainer.fit(module, data)

    optimizer = trainer.optimizers[0]
    assert isinstance(optimizer, EMAOptimizer)
    checkpoint = torch.load(tmp_path / "ema_test.ckpt", map_location="cpu")
    saved = [checkpoint["state_dict"][name] for name, _ in module.named_parameters()]
    current = [param.detach().cpu() for param in module.parameters()]
    ema = [param.cpu() for param in optimizer.ema_params]

    assert all(torch.equal(s, e) for s, e in zip(saved, ema))
    assert not all(torch.equal(s, c) for s, c in zip(saved, current))
