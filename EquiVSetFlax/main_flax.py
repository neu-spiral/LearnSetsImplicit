# Standard libraries
import os
import sys
import argparse
from typing import Any, Sequence, Optional, Tuple, Iterator, Dict, Callable, Union
import json
import time
from tqdm.auto import tqdm
import numpy as np
from copy import copy
from glob import glob
from collections import defaultdict

# JAX/Flax
# If you run this code on Colab, remember to install flax and optax
# !pip install --quiet --upgrade flax optax
import jax
import jax.numpy as jnp
from jax import random
from flax import linen as nn
from flax.training import train_state, checkpoints
import optax

# PyTorch for data loading
import torch
import torch.utils.data as data
from torchvision import transforms

# Logging with Tensorboard or Weights and Biases
# If you run this code on Colab, remember to install pytorch_lightning
# !pip install --quiet --upgrade pytorch_lightning
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger

from model.EquiVSet_trainer import EquiVSetTrainer
from data_loader import TwoMoons

if __name__ == "__main__":
    params = {'data_name': 'moons', 'root_path': './', 'v_size': 100, 's_size': 10, 'num_layers': 2, 'batch_size': 128,
              'lr': 0.0001, 'weight_decay': 1e-5, 'init': 0.05, 'clip': 10, 'epochs': 100, 'num_bad_epochs': 6,
              'num_runs': 1, 'num_workers': 2, 'seed': 50971, 'mode': 'diffMF', 'RNN_steps': 1, 'num_samples': 5,
              'rank': 5, 'tau': 0.1, 'neg_num': 1}

    # parser = argparse.ArgumentParser()
    # params = [f"--{k}={v}" for k, v in params.items()]
    # params = parser.parse_args()

    # get data
    data = TwoMoons(params)
    CHECKPOINT_PATH = '../checkpoints/'
    batch_size = params['batch_size']
    num_workers = params['num_workers']


    def transform(x):
        return x.numpy().astype(np.float32)


    def tensor_to_numpy(x):
        x = np.array(x, dtype=np.float32)
        return x


    train_loader, val_loader, test_loader = data.get_loaders(batch_size, num_workers, transform=tensor_to_numpy)
    # print(next(iter(train_loader)))

    trainer = EquiVSetTrainer(params=params,
                              dim_feature=256,
                              optimizer_hparams={'lr': 4e-3},
                              logger_params={'base_log_dir': CHECKPOINT_PATH},
                              exmp_input=next(iter(train_loader)),
                              check_val_every_n_epoch=5)

    metrics = trainer.train_model(train_loader,
                                  val_loader,
                                  test_loader=test_loader,
                                  num_epochs=50)

    print(f'Training loss: {metrics["train/loss"]}')
    print(f'Validation loss: {metrics["val/loss"]}')
    print(f'Test loss: {metrics["test/loss"]}')