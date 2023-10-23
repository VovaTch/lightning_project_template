from typing import Any

import torch

from loss.aggregators import build_loss_aggregator
from loss.components import build_classification_loss


def test_basic_classification_loss() -> None:
    loss_cfg = {"type": "basic_loss", "weight": 3.0, "basic_loss": "ce"}
    loss_component = build_classification_loss("test_loss", loss_cfg)
    estimation = {"pred_logits": torch.randn((4, 10))}
    target = {"class": torch.randint(0, 10, (4,))}
    loss = loss_component(estimation, target)
    assert list(loss.size()) == []


def test_loss_aggregator(cfg: dict[str, Any]) -> None:
    loss_aggregator = build_loss_aggregator(cfg)
    estimation = {"pred_logits": torch.randn((4, 10))}
    target = {"class": torch.randint(0, 10, (4,))}
    loss = loss_aggregator(estimation, target)
    assert list(loss.total.size()) == []
    assert len(loss.individuals) == 1
