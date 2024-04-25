import torch


def transparent(x: torch.Tensor) -> torch.Tensor:
    return x


def tanh(x: torch.Tensor) -> torch.Tensor:
    return torch.tanh(x)


def sigmoid(x: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(x)
