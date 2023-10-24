from dataclasses import dataclass
from typing import Any


@dataclass
class LearningParameters:
    # Name
    model_name: str

    # Learning settings
    learning_rate: float
    weight_decay: float
    batch_size: int
    epochs: int
    beta_ema: float
    gradient_clip: float
    save_path: str
    eval_split_factor: float
    amp: bool
    num_devices: int = 1
    num_workers: int = 0

    # Scheduler settings
    loss_monitor: str = "step"
    interval: str = "training_total_loss"
    frequency: int = 1


def parse_learning_parameters_from_cfg(cfg: dict[str, Any]) -> LearningParameters:
    """
    Utility method to parse learning parameters from a configuration dictionary

    Args:
        cfg (dict[str, Any]): configuration dictionary

    Returns:
        LearningParameters: Learning parameters object
    """
    learning_params = cfg["learn"]
    return LearningParameters(
        model_name=cfg["model_name"],
        learning_rate=learning_params["learning_rate"],
        weight_decay=learning_params["weight_decay"],
        batch_size=learning_params["batch_size"],
        epochs=learning_params["epochs"],
        beta_ema=learning_params["beta_ema"],
        gradient_clip=learning_params["gradient_clip"],
        save_path=learning_params["save_path"],
        eval_split_factor=learning_params["eval_split_factor"],
        amp=learning_params["amp"],
        loss_monitor=learning_params["scheduler"]["loss_monitor"],
        interval=learning_params["scheduler"]["interval"],
        frequency=learning_params["scheduler"]["frequency"],
    )
