from typing import Any

import yaml


def load_config(path: str) -> dict[str, Any]:
    """
    Loads a yaml configuration file from path into a dictionary

    Args:
        path (str): File path

    Returns:
        dict[str, Any]: Configuration dictionary
    """
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg
