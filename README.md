# Lightning Project Template

This project template implements a simple MNIST classification code with a fully connected network. 
Uses:
* Lightning
* Hydra

## Install

- Have [uv](https://docs.astral.sh/uv/) installed in the system.
- Clone this repository.
- Run `uv sync` (includes the dev group: pytest, black, pyright, notebook deps).

## Run

- Training: `uv run python -m scripts.train`.
- Testing loss on the test set: `uv run python -m scripts.test`.
- Override any config value Hydra-style, e.g. `uv run python -m scripts.train learning.epochs=5 learning.batch_size=64`.
- **notebooks/run_network.ipynb** is a notebook for visualizing network results.

## Develop

- Tests: `uv run pytest`.
- Formatting: `uv run black .`.
- Type checking: `uv run pyright`.
