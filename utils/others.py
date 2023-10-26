from typing import Any, Callable

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


def register_builder(registry: dict[str, Callable], key: str) -> Callable:
    """
    Decorator to register a builder function into a key that can be called from a configuration
    file.

    Args:
        registry (dict[str, Callable]): Registry dictionary
        key (str): New key

    Returns:
        Callable: The wrapped function
    """

    def wrapper(func: Callable) -> Callable:
        registry.update({key: func})
        return func

    return wrapper


class RegisterClass:
    """
    Decorator to register a class into a key that can be called from a configuration
    """

    def __init__(self, registry: dict[str, Callable], key: str) -> None:
        """
        Constructor

        Args:
            registry (dict[str, Callable]): Registry dictionary
            key (str): New key
        """
        self.registry = registry
        self.key = key

    def __call__(self, cls: type) -> type:
        """
        Wrapper call

        Args:
            cls (type): class to insert into a registry

        Returns:
            type: Class after registry
        """
        self.registry.update({self.key: cls})
        return cls
