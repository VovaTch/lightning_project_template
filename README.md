# Lightning Project Template

This project template implements a simple MNIST classification code with a fully connected network.

## Install

- Install the Pytorch package corresponding to your CUDA version.
- Run in the terminal `pip install -r requirements.txt`

## Run:

- For training: `python -m scripts.train`.
- For testing loss on test set: `python -m scripts.test`
- For placing the trained weights in a more accessible folder **weights/** run `python -m scripts.pull_last_weights`, use the `--del` flag to delete the checkpoint history to reduce clutter and save space.
- **notebooks/run_network.ipynb** is a notebook for visualizing network results.
