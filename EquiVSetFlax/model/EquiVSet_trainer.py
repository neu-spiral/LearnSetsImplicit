# Standard libraries
import os
import sys
from typing import Any, Sequence, Optional, Tuple, Iterator, Dict, Callable, Union, List
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
from model.set_functions_flax import SetFunction, RecNet, MC_sampling
from utils.flax_evaluation import compute_metrics
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
            # V_set, S_set, neg_S_set = batch
            step_metrics = self.eval_step(self.state, batch)  # assume this returns jc
            # batch_size = batch[0].shape[0] if isinstance(batch, (list, tuple)) else batch.shape[0]
            for key in step_metrics:  # metrics[key] = jc_list
                if key in metrics:
                    metrics[key].append(step_metrics[key])
                else:
                    metrics[key] = [step_metrics[key]]
            # num_elements += batch_size
        metrics = {(log_prefix + key): 100 * np.array(jnp.concatenate(metrics[key], axis=0).mean(0)) for key in metrics}  # convert to numpy
        return metrics

    def create_functions(self):
        def entropy_loss(params, batch):
            V, S, neg_S = batch
            # jax.debug.print("V is {V}", V=V)
            # jax.debug.print("S is {S}", S=S)
            # jax.debug.print("neg_S is {neg_S}", neg_S=neg_S)
            return self.model.apply({'params': params}, V, S, neg_S)

        def inference(state, V, bs):
            # V, S, neg_S = batch
            bs, vs = V.shape[:2]
            q = .5 * jnp.ones((bs, vs))

            for i in range(self.model_hparams['params']['RNN_steps']):
                sample_matrix_1, sample_matrix_0 = MC_sampling(q, self.model_hparams['params']['num_samples'])
                q = self.model.apply({'params': state.params}, V, sample_matrix_1, sample_matrix_0, method='mean_field_iteration')
            return q

        def train_step(state, batch):
            loss_fn = lambda params: entropy_loss(params, batch)
            loss, grads = jax.value_and_grad(loss_fn)(state.params)
            # jax.debug.print("loss is {loss}\n", loss=loss)
            # jax.debug.print("grads is {grads}\n", grads=grads)
            state = state.apply_gradients(grads=grads)
            metrics = {'loss': loss}
            return state, metrics

        def eval_step(state, batch):
            # loss = entropy_loss(state.params, batch)
            V_set, S_set, neg_S_set = batch

            q = inference(state, V_set, S_set.shape[0])
            # idx = jnp.argpartition(q, -S_set.shape[-1], axis=1)  # [-S_set.shape[-1]:]
            _, idx = jax.lax.top_k(q, S_set.shape[-1])
            # call(lambda x: print(f"shape[0] {x}"), S_set.shape[0])  # 128
            # call(lambda x: print(f"shape[-1] {x}"), S_set.shape[-1])  # 100
            # call(lambda x: print(f"len idx {x}"), len(idx))  # 100

            pre_list = []
            # s_size = jnp.sum(S_set[0]).astype(int)  # needs s_size as input
            for i in range(len(idx)):
                pre_mask = jnp.zeros([S_set.shape[-1]])
                # print(np.array(jnp.sum(S_set[i])))
                # call(lambda x: print(f"sum is {x}"), jnp.sum(S_set[i]))
                ids = idx[i][:self.model_hparams['params']['s_size']]  # this needs static slicing
                pre_mask = pre_mask.at[ids].set(1)
                pre_list.append(jnp.expand_dims(pre_mask, axis=0))
            pre_mask = jnp.concatenate(pre_list, axis=0)
            true_mask = S_set

            intersection = true_mask * pre_mask
            union = true_mask + pre_mask - intersection
            jc = intersection.sum(axis=-1) / union.sum(axis=-1)
            return {'jaccard': jc}  # loss

        return train_step, eval_step
