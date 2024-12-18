# Standard libraries
import os
import sys
from typing import Any, Optional, Iterator, Dict, Callable, Union, List
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
from functools import partial
import optax

# PyTorch for data loading
import torch
import torch.utils.data as data

# Logging with Tensorboard or Weights and Biases
# If you run this code on Colab, remember to install pytorch_lightning
# !pip install --quiet --upgrade pytorch_lightning
# from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger

from model.trainer_module import TrainerModule
from model.set_functions_flax import SetFunction, RecNet  # , MFVI
from jax.experimental.host_callback import call


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

    def eval_model(self,
                   data_loader: Iterator,
                   log_prefix: Optional[str] = '') -> Dict[str, Any]:
        """
        Evaluates the model on a dataset.

        Args:
          data_loader: Data loader of the dataset to evaluate on.
          log_prefix: Prefix to add to all metrics (e.g. 'val/' or 'test/')

        Returns:
          A dictionary of the evaluation metrics, averaged over data points
          in the dataset.
        """
        # Test model on all images of a data loader and return avg loss
        metrics = defaultdict(List)  # defaultdict(float)
        # jc_list = []
        for batch in data_loader:
            if len(batch) == 3:
                V_set, S_set, _ = batch
            elif len(batch) == 2:
                V_set, S_set = batch
            step_metrics = self.eval_step(self.state, V_set, S_set)  # assume this returns jc
            for key in step_metrics:  # metrics[key] = jc_list
                if key in metrics:
                    metrics[key].append(step_metrics[key])
                else:
                    metrics[key] = [step_metrics[key]]
            # num_elements += batch_size

        # convert to numpy
        metrics = {(log_prefix + key): 1 * np.array(jnp.concatenate(metrics[key], axis=0).mean(0)) for key in metrics}
        return metrics

    def create_functions(self):
        def entropy_loss(params, batch):
            V, S, neg_S = batch
            return self.model.apply({'params': params}, V, S, neg_S)

        def inference(state, V):
            # V, S, neg_S = batch
            if self.model_hparams['params'].data_name == 'bindingdb':
                bs = self.model_hparams['params'].batch_size
                vs = self.model_hparams['params'].v_size
            else:
                bs, vs = V.shape[:2]
                if self.model_hparams['params'].data_name == 'celeba':
                    bs = int(bs / 8)
                    vs = self.model_hparams['params'].v_size
            q = .5 * jnp.ones((bs, vs))
            # print(state.params)

            for i in range(self.model_hparams['params'].RNN_steps):
                model = self.bind_model()
                mfvi = model.mfvi.bind({'params': state.params['fixed_point_layer']['mfvi_params']['params']})
                q = mfvi(q, V)
            return q

        def train_step(state, batch):
            loss_fn = lambda params: entropy_loss(params, batch)
            loss, grads = jax.value_and_grad(loss_fn)(state.params)
            state = state.apply_gradients(grads=grads)
            metrics = {'loss': loss}
            return state, metrics

        def eval_step(state, V_set, S_set):
            q = inference(state, V_set)
            _, idx = jax.lax.top_k(q, S_set.shape[-1])

            pre_list = []
            for i in range(len(idx)):
                pre_mask = jnp.zeros([S_set.shape[-1]])
                s_size = jnp.sum(S_set[i]).astype(int)  # needs s_size as input
                mask = jnp.where(jnp.arange(S_set.shape[-1]) < s_size, True, self.model_hparams['params'].v_size + 1)
                ids = (idx[i] + 1) * mask - 1
                pre_mask = pre_mask.at[ids].set(1)  # , mode='drop')
                pre_list.append(jnp.expand_dims(pre_mask, axis=0))
            pre_mask = jnp.concatenate(pre_list, axis=0)
            true_mask = S_set

            intersection = true_mask * pre_mask
            union = true_mask + pre_mask - intersection
            jc = intersection.sum(axis=-1) / union.sum(axis=-1)
            jc = 100 * jc

            return {'jaccard': jc}  # loss

        return train_step, eval_step
