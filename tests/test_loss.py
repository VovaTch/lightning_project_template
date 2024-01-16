from omegaconf import DictConfig, OmegaConf
import torch

from loss.aggregators import build_loss_aggregator
from loss.components import build_classification_loss


def test_basic_classification_loss() -> None:
    """
    Test case for basic classification loss.

    This test case creates a basic classification loss configuration using OmegaConf.
    It builds a classification loss component based on the configuration.
    Then, it generates random prediction logits and target classes.
    Finally, it computes the loss using the loss component and asserts that the size of the loss tensor is empty.

    Returns:
        None
    """
    loss_cfg = OmegaConf.create(
        {"type": "basic_loss", "weight": 3.0, "basic_loss": "ce"}
    )
    loss_component = build_classification_loss("test_loss", loss_cfg)
    estimation = {"pred_logits": torch.randn((4, 10))}
    target = {"class": torch.randint(0, 10, (4,))}
    loss = loss_component(estimation, target)
    assert list(loss.size()) == []


def test_loss_aggregator(cfg: DictConfig) -> None:
    """
    Test the loss aggregator function.

    Args:
        cfg (DictConfig): The configuration dictionary.

    Returns:
        None
    """
    loss_aggregator = build_loss_aggregator(cfg)
    estimation = {"pred_logits": torch.randn((4, 10))}
    target = {"class": torch.randint(0, 10, (4,))}
    loss = loss_aggregator(estimation, target)
    assert list(loss.total.size()) == []
    assert len(loss.individuals) == 2
