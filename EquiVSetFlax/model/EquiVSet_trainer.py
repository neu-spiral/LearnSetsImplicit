# Standard libraries
import os
import sys
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

# Logging with Tensorboard or Weights and Biases
# If you run this code on Colab, remember to install pytorch_lightning
# !pip install --quiet --upgrade pytorch_lightning
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger

from model.trainer_module import TrainerModule
from model.set_functions_flax import SetFunction, RecNet


class EquiVSetTrainer(TrainerModule):

    def __init__(self,
                 params: Dict[str, Any],
                 dim_feature: int,
                 **kwargs):
        super().__init__(model_class=SetFunction,
                         model_hparams={
                             'params': params,
                             'dim_feature': dim_feature
                         },
                         **kwargs)

    def create_functions(self):
        def entropy_loss(params, batch):
            V, S, neg_S = batch
            return self.model.apply({'params': params}, V, S, neg_S)

        def train_step(state, batch):
            loss_fn = lambda params: entropy_loss(params, batch)
            loss, grads = jax.value_and_grad(loss_fn)(state.params)
            state = state.apply_gradients(grads=grads)
            metrics = {'loss': loss}
            return state, metrics

        def eval_step(state, batch):
            loss = entropy_loss(state.params, batch)
            return {'loss': loss}

        return train_step, eval_step
