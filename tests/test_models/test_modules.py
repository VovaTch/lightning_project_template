from __future__ import annotations

from functools import partial
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from loss.aggregators import LossOutput, WeightedSumAggregator
from loss.components import BasicClassificationLoss
from models.base import load_inner_model_state_dict
from models.models import FCN
from models.modules import MnistClassifierModule
from utils.learning import LearningParameters


@pytest.fixture
def cls_loss_aggregator() -> WeightedSumAggregator:
    cls_loss_component = BasicClassificationLoss(
        "cls_loss", 1.0, torch.nn.CrossEntropyLoss()
    )
    return WeightedSumAggregator([cls_loss_component])


@pytest.fixture
def mnist_classifier_module(
    cls_loss_aggregator: WeightedSumAggregator,
) -> MnistClassifierModule:
    # Create a sample model, learning parameters, and other required objects
    model = FCN()
    learning_params = LearningParameters("testing")
    transforms = nn.Sequential()
    optimizers = None
    schedulers = None

    # Create an instance of the MnistClassifierModule
    module = MnistClassifierModule(
        model=model,
        learning_params=learning_params,
        transforms=transforms,
        loss_aggregator=cls_loss_aggregator,
        optimizers=optimizers,
        schedulers=schedulers,
    )
    return module


def test_forward(mnist_classifier_module: MnistClassifierModule) -> None:
    x = {"images": torch.randn(10, 28, 28)}
    output = mnist_classifier_module.forward(x)
    assert output["logits"].shape == (10, 10)


def test_step(mnist_classifier_module: MnistClassifierModule) -> None:
    batch = {"images": torch.randn(10, 28, 28), "class": torch.randint(0, 10, (10,))}
    phase = "train"
    loss = mnist_classifier_module.step(batch, phase)
    assert loss is None or isinstance(loss, torch.Tensor)


def test_log_loss(mnist_classifier_module: MnistClassifierModule) -> None:
    loss = LossOutput(
        torch.tensor(0.25),
        {"component1": torch.tensor(0.1), "component2": torch.tensor(0.15)},
    )
    phase = "train"
    total_loss = mnist_classifier_module.log_loss(loss, phase)
    assert isinstance(total_loss, torch.Tensor)


def test_configure_optimizers_default(
    mnist_classifier_module: MnistClassifierModule,
) -> None:
    optimizers = mnist_classifier_module.configure_optimizers()
    assert isinstance(optimizers, list)
    assert isinstance(optimizers[0], torch.optim.AdamW)


def test_configure_optimizers_with_scheduler(
    cls_loss_aggregator: WeightedSumAggregator,
) -> None:
    learning_params = LearningParameters("testing", interval="epoch", frequency=2)
    module = MnistClassifierModule(
        model=FCN(),
        learning_params=learning_params,
        loss_aggregator=cls_loss_aggregator,
        optimizers=partial(torch.optim.SGD, lr=0.1),
        schedulers=partial(torch.optim.lr_scheduler.StepLR, step_size=1),
    )
    configs = module.configure_optimizers()
    assert isinstance(configs, list)
    config = configs[0]
    assert isinstance(config, dict) and "lr_scheduler" in config
    assert isinstance(config["optimizer"], torch.optim.SGD)
    scheduler_config = config["lr_scheduler"]
    assert isinstance(scheduler_config, dict)
    assert isinstance(scheduler_config["scheduler"], torch.optim.lr_scheduler.StepLR)
    assert scheduler_config.get("interval") == "epoch"
    assert scheduler_config.get("monitor") == learning_params.loss_monitor
    assert scheduler_config.get("frequency") == 2


def test_scheduler_count_mismatch(cls_loss_aggregator: WeightedSumAggregator) -> None:
    sgd = partial(torch.optim.SGD, lr=0.1)
    step_lr = partial(torch.optim.lr_scheduler.StepLR, step_size=1)
    module = MnistClassifierModule(
        model=FCN(),
        learning_params=LearningParameters("testing"),
        optimizers=[sgd],
        schedulers=[step_lr, step_lr],
    )
    with pytest.raises(ValueError):
        module.configure_optimizers()


def test_load_missing_checkpoint_raises(
    mnist_classifier_module: MnistClassifierModule, tmp_path: Path
) -> None:
    with pytest.raises(FileNotFoundError):
        load_inner_model_state_dict(mnist_classifier_module, str(tmp_path / "x.ckpt"))


def test_load_compiled_checkpoint(
    mnist_classifier_module: MnistClassifierModule, tmp_path: Path
) -> None:
    source = FCN()
    state_dict = {f"model._orig_mod.{k}": v for k, v in source.state_dict().items()}
    path = tmp_path / "compiled.ckpt"
    torch.save({"state_dict": state_dict}, path)

    load_inner_model_state_dict(mnist_classifier_module, str(path))
    for key, value in source.state_dict().items():
        assert torch.equal(mnist_classifier_module.model.state_dict()[key], value)
