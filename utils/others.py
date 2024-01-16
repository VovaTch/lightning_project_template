from typing import Callable


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
