from __future__ import annotations

import contextlib
import copy
import threading
from typing import Any, Callable, Iterable, Iterator, Sequence, overload

import lightning as L
import torch
from lightning.pytorch import Callback
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.fabric.utilities.exceptions import MisconfigurationException


class EMA(Callback):
    """
    Implements Exponential Moving Averaging (EMA).

    When training a model, this callback will maintain moving averages of the trained parameters.
    When evaluating, we use the moving averages copy of the trained parameters.
    Pair with `EMAModelCheckpoint` to save the EMA weights.

    Args:
    *   decay (float): The exponential decay used when calculating the moving average. Has to be between 0-1.
    *   validate_original_weights (bool): Validate the original weights, as opposed to the EMA weights.
    *   every_n_steps (int): Apply EMA every N steps.
    *   cpu_offload (bool): Offload weights to CPU.
    """

    def __init__(
        self,
        decay: float,
        validate_original_weights: bool = False,
        every_n_steps: int = 1,
        cpu_offload: bool = False,
    ) -> None:
        if not (0 <= decay <= 1):
            raise MisconfigurationException("EMA decay value must be between 0 and 1")
        self.decay = decay
        self.validate_original_weights = validate_original_weights
        self.every_n_steps = every_n_steps
        self.cpu_offload = cpu_offload

    def on_fit_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """
        Wraps the trainer's optimizers with EMAOptimizer.

        Args:
        *   trainer (L.Trainer): The trainer.
        *   pl_module (L.LightningModule): The trained module.
        """
        device = pl_module.device if not self.cpu_offload else torch.device("cpu")
        trainer.optimizers = [
            EMAOptimizer(
                optim,
                device=device,
                decay=self.decay,
                every_n_steps=self.every_n_steps,
                current_step=trainer.global_step,
            )
            for optim in trainer.optimizers
            if not isinstance(optim, EMAOptimizer)
        ]

    def on_validation_start(
        self, trainer: L.Trainer, pl_module: L.LightningModule
    ) -> None:
        """
        Swaps in the EMA weights on validation start, if evaluating on them.

        Args:
        *   trainer (L.Trainer): The trainer.
        *   pl_module (L.LightningModule): The evaluated module.
        """
        if self._should_validate_ema_weights(trainer):
            self.swap_model_weights(trainer)

    def on_validation_end(
        self, trainer: L.Trainer, pl_module: L.LightningModule
    ) -> None:
        """
        Swaps in the EMA weights on validation end (swaps back), if evaluating on them.

        Args:
        *   trainer (L.Trainer): The trainer.
        *   pl_module (L.LightningModule): The evaluated module.
        """
        if self._should_validate_ema_weights(trainer):
            self.swap_model_weights(trainer)

    def on_test_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """
        Swaps in the EMA weights on test start, if evaluating on them.

        Args:
        *   trainer (L.Trainer): The trainer.
        *   pl_module (L.LightningModule): The evaluated module.
        """
        if self._should_validate_ema_weights(trainer):
            self.swap_model_weights(trainer)

    def on_test_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """
        Swaps in the EMA weights on test end (swaps back), if evaluating on them.

        Args:
        *   trainer (L.Trainer): The trainer.
        *   pl_module (L.LightningModule): The evaluated module.
        """
        if self._should_validate_ema_weights(trainer):
            self.swap_model_weights(trainer)

    def _should_validate_ema_weights(self, trainer: L.Trainer) -> bool:
        """
        Args:
        *   trainer (L.Trainer): The trainer.

        Returns:
        *   bool: Whether evaluation should run on the EMA weights.
        """
        return not self.validate_original_weights and self._ema_initialized(trainer)

    def _ema_initialized(self, trainer: L.Trainer) -> bool:
        """
        Args:
        *   trainer (L.Trainer): The trainer.

        Returns:
        *   bool: Whether any of the trainer's optimizers is an EMAOptimizer.
        """
        return any(
            isinstance(optimizer, EMAOptimizer) for optimizer in trainer.optimizers
        )

    def _ema_optimizers(self, trainer: L.Trainer) -> list[EMAOptimizer]:
        """
        Args:
        *   trainer (L.Trainer): The trainer.

        Raises:
        *   TypeError: A trainer optimizer is not an EMAOptimizer.

        Returns:
        *   list[EMAOptimizer]: The trainer's optimizers.
        """
        optimizers = trainer.optimizers
        if not all(isinstance(optimizer, EMAOptimizer) for optimizer in optimizers):
            raise TypeError("All trainer optimizers must be EMAOptimizer instances.")
        return [opt for opt in optimizers if isinstance(opt, EMAOptimizer)]

    def swap_model_weights(
        self, trainer: L.Trainer, saving_ema_model: bool = False
    ) -> None:
        """
        Swaps the model weights with the EMA weights, in place.

        Args:
        *   trainer (L.Trainer): The trainer.
        *   saving_ema_model (bool): Whether the swap is for saving the EMA model.
        """
        for optimizer in self._ema_optimizers(trainer):
            optimizer.switch_main_parameter_weights(saving_ema_model)

    @contextlib.contextmanager
    def save_ema_model(self, trainer: L.Trainer) -> Iterator[None]:
        """
        Swaps in the EMA weights for the context duration, e.g. for saving.

        Args:
        *   trainer (L.Trainer): The trainer.

        Returns:
        *   Iterator[None]: Context manager.
        """
        self.swap_model_weights(trainer, saving_ema_model=True)
        try:
            yield
        finally:
            self.swap_model_weights(trainer, saving_ema_model=False)

    @contextlib.contextmanager
    def save_original_optimizer_state(self, trainer: L.Trainer) -> Iterator[None]:
        """
        Makes the EMA optimizers return the wrapped optimizer state in the context.

        Args:
        *   trainer (L.Trainer): The trainer.

        Returns:
        *   Iterator[None]: Context manager.
        """
        optimizers = self._ema_optimizers(trainer)
        for optimizer in optimizers:
            optimizer.save_original_optimizer_state = True
        try:
            yield
        finally:
            for optimizer in optimizers:
                optimizer.save_original_optimizer_state = False


class EMAModelCheckpoint(ModelCheckpoint):
    """
    ModelCheckpoint that saves the EMA weights whenever validation runs on them,
    so the saved weights match the monitored metric. Behaves like ModelCheckpoint
    when the EMA callback is inactive.

    Args:
    *   ema (EMA): The EMA callback registered on the same trainer.
    *   **kwargs (Any): ModelCheckpoint arguments.
    """

    def __init__(self, ema: EMA, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.ema = ema

    def _save_checkpoint(self, trainer: L.Trainer, filepath: str) -> None:
        """
        Saves the checkpoint, swapping in the EMA weights for the save if needed.

        Args:
        *   trainer (L.Trainer): The trainer.
        *   filepath (str): Checkpoint destination.
        """
        if not self.ema._should_validate_ema_weights(trainer):
            super()._save_checkpoint(trainer, filepath)
            return
        with self.ema.save_ema_model(trainer):
            super()._save_checkpoint(trainer, filepath)


@torch.no_grad()
def ema_update(
    ema_params: Sequence[torch.Tensor],
    current_params: Sequence[torch.Tensor],
    decay: float,
) -> None:
    """
    In-place EMA update: ema = decay * ema + (1 - decay) * current.

    Args:
    *   ema_params (Sequence[torch.Tensor]): EMA tensors, updated in place.
    *   current_params (Sequence[torch.Tensor]): Current parameter tensors.
    *   decay (float): Decay factor.
    """
    for ema_param, current_param in zip(ema_params, current_params):
        ema_param.lerp_(current_param, 1.0 - decay)


def run_ema_update_cpu(
    ema_params: Sequence[torch.Tensor],
    current_params: Sequence[torch.Tensor],
    decay: float,
    pre_sync_stream: torch.Stream | None = None,
) -> None:
    """
    EMA update for CPU-offloaded weights, waits for the copy stream first.

    Args:
    *   ema_params (Sequence[torch.Tensor]): EMA tensors, updated in place.
    *   current_params (Sequence[torch.Tensor]): Current parameter tensors.
    *   decay (float): Decay factor.
    *   pre_sync_stream (torch.Stream | None): Stream to synchronize before updating.
    """
    if pre_sync_stream is not None:
        pre_sync_stream.synchronize()

    ema_update(ema_params, current_params, decay)


class EMAOptimizer(torch.optim.Optimizer):
    r"""
    EMAOptimizer is a wrapper for torch.optim.Optimizer that computes
    Exponential Moving Average of parameters registered in the optimizer.

    EMA parameters are automatically updated after every step of the optimizer
    with the following formula:

        ema_weight = decay * ema_weight + (1 - decay) * training_weight

    To access EMA parameters, use ``swap_ema_weights()`` context manager to
    perform a temporary in-place swap of regular parameters with EMA
    parameters.

    Notes:
        - EMAOptimizer is not compatible with APEX AMP O2.

    Example:
        model = Model().to(device)
        opt = torch.optim.Adam(model.parameters())

        opt = EMAOptimizer(opt, device, 0.9999)

        for epoch in range(epochs):
            training_loop(model, opt)

            regular_eval_accuracy = evaluate(model)

            with opt.swap_ema_weights():
                ema_eval_accuracy = evaluate(model)

    Args:
    *   optimizer (torch.optim.Optimizer): Optimizer to wrap.
    *   device (torch.device): Device for EMA parameters.
    *   decay (float): Decay factor.
    *   every_n_steps (int): Apply EMA every N steps.
    *   current_step (int): Starting step count.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        decay: float = 0.9999,
        every_n_steps: int = 1,
        current_step: int = 0,
    ) -> None:
        self.optimizer = optimizer
        self.decay = decay
        self.device = device
        self.current_step = current_step
        self.every_n_steps = every_n_steps
        self.save_original_optimizer_state = False

        self.first_iteration = True
        self.rebuild_ema_params = True
        self.stream: torch.Stream | None = None
        self.thread: threading.Thread | None = None

        self.ema_params: tuple[torch.Tensor, ...] = ()
        self.in_saving_ema_model_context = False

    def all_parameters(self) -> Iterable[torch.Tensor]:
        """
        Returns:
        *   Iterable[torch.Tensor]: All parameters of the wrapped optimizer.
        """
        return (param for group in self.param_groups for param in group["params"])

    @overload
    def step(self, closure: None = None) -> None: ...

    @overload
    def step(self, closure: Callable[[], float]) -> float: ...

    def step(self, closure: Callable[[], float] | None = None) -> float | None:
        """
        Steps the wrapped optimizer, then updates the EMA weights.

        Args:
        *   closure (Callable[[], float] | None): Loss closure.

        Returns:
        *   float | None: Closure loss, if any.
        """
        self.join()

        if self.first_iteration:
            if any(p.is_cuda for p in self.all_parameters()):
                self.stream = torch.cuda.Stream()

            self.first_iteration = False

        if self.rebuild_ema_params:
            opt_params = list(self.all_parameters())

            self.ema_params += tuple(
                copy.deepcopy(param.data.detach()).to(self.device)
                for param in opt_params[len(self.ema_params) :]
            )
            self.rebuild_ema_params = False

        loss = self.optimizer.step(closure)

        if self._should_update_at_step():
            self.update()
        self.current_step += 1
        return loss

    def _should_update_at_step(self) -> bool:
        """
        Returns:
        *   bool: Whether the EMA should update at the current step.
        """
        return self.current_step % self.every_n_steps == 0

    @torch.no_grad()
    def update(self) -> None:
        """
        Updates the EMA weights, asynchronously on a side stream or CPU thread.
        """
        stream_context: contextlib.AbstractContextManager[Any] = (
            contextlib.nullcontext()
        )
        if self.stream is not None:
            self.stream.wait_stream(torch.cuda.current_stream())
            stream_context = self.stream

        with stream_context:
            current_model_state = tuple(
                param.data.to(self.device, non_blocking=True)
                for param in self.all_parameters()
            )

            if self.device.type == "cuda":
                ema_update(self.ema_params, current_model_state, self.decay)

        if self.device.type == "cpu":
            self.thread = threading.Thread(
                target=run_ema_update_cpu,
                args=(
                    self.ema_params,
                    current_model_state,
                    self.decay,
                    self.stream,
                ),
            )
            self.thread.start()

    def swap_tensors(self, tensor1: torch.Tensor, tensor2: torch.Tensor) -> None:
        """
        Swaps two tensors' contents in place.

        Args:
        *   tensor1 (torch.Tensor): First tensor.
        *   tensor2 (torch.Tensor): Second tensor.
        """
        tmp = torch.empty_like(tensor1)
        tmp.copy_(tensor1)
        tensor1.copy_(tensor2)
        tensor2.copy_(tmp)

    def switch_main_parameter_weights(self, saving_ema_model: bool = False) -> None:
        """
        Swaps the model parameters with the EMA parameters, in place.

        Args:
        *   saving_ema_model (bool): Whether the swap is for saving the EMA model.
        """
        self.join()
        self.in_saving_ema_model_context = saving_ema_model
        for param, ema_param in zip(self.all_parameters(), self.ema_params):
            self.swap_tensors(param.data, ema_param)

    @contextlib.contextmanager
    def swap_ema_weights(self, enabled: bool = True) -> Iterator[None]:
        """
        A context manager to in-place swap regular parameters with EMA
        parameters. It swaps back to the original regular parameters on exit.

        Args:
        *   enabled (bool): Whether the swap should be performed.

        Returns:
        *   Iterator[None]: Context manager.
        """
        if enabled:
            self.switch_main_parameter_weights()
        try:
            yield
        finally:
            if enabled:
                self.switch_main_parameter_weights()

    def __getattr__(self, name: str) -> Any:
        """
        Delegates missing attributes to the wrapped optimizer.

        Args:
        *   name (str): Attribute name.

        Returns:
        *   Any: The wrapped optimizer's attribute.
        """
        return getattr(self.optimizer, name)

    def join(self) -> None:
        """
        Waits for any pending EMA update.
        """
        if self.stream is not None:
            self.stream.synchronize()

        if self.thread is not None:
            self.thread.join()

    def state_dict(self) -> dict[str, Any]:
        """
        Returns:
        *   dict[str, Any]: Wrapped optimizer state plus EMA state.
        """
        self.join()

        if self.save_original_optimizer_state:
            return self.optimizer.state_dict()

        # In the EMA-saving context the EMA weights live in the module.
        ema_params = (
            self.ema_params
            if not self.in_saving_ema_model_context
            else list(self.all_parameters())
        )
        return {
            "opt": self.optimizer.state_dict(),
            "ema": ema_params,
            "current_step": self.current_step,
            "decay": self.decay,
            "every_n_steps": self.every_n_steps,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """
        Args:
        *   state_dict (dict[str, Any]): State from `state_dict`.
        """
        self.join()

        self.optimizer.load_state_dict(state_dict["opt"])
        self.ema_params = tuple(
            param.to(self.device) for param in copy.deepcopy(state_dict["ema"])
        )
        self.current_step = state_dict["current_step"]
        self.decay = state_dict["decay"]
        self.every_n_steps = state_dict["every_n_steps"]
        self.rebuild_ema_params = False

    def add_param_group(self, param_group: dict[str, Any]) -> None:
        """
        Args:
        *   param_group (dict[str, Any]): Parameter group to add.
        """
        self.optimizer.add_param_group(param_group)
        self.rebuild_ema_params = True
