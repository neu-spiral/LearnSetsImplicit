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
import psutil
import pandas as pd
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
# from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger

from model.EquiVSet_trainer import EquiVSetTrainer
from data_loader import TwoMoons, GaussianMixture, Amazon, CelebA, SetPDBBind, SetBindingDB
import utils.config as config_file

# jax.config.update("jax_debug_nans", True)  # stops execution when nan occurs
jax.config.update("jax_enable_x64", True)  # solves the nan value issue when calculating q, hence loss


def get_data(params):
    data_name = params.data_name
    if data_name == 'moons':
        return TwoMoons(params)
    elif data_name == 'gaussian':
        return GaussianMixture(params)
    elif data_name == 'amazon':
        return Amazon(params)
    elif data_name == 'celeba':
        return CelebA(params)
    elif data_name == 'pdbbind':
        return SetPDBBind(params)
    elif data_name == 'bindingdb':
        return SetBindingDB(params)
    else:
        raise ValueError(
            "Invalid data_name. Supported options are: 'moons', 'gaussian', 'amazon', 'celeba', 'pdbbind', 'bindingdb'")


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manual_params', action='store_true',
                        help='Flag to indicate if manual parameters will be provided')
    parser.add_argument('--data_name', type=str, default='moons',
                        choices=['moons', 'gaussian', 'amazon', 'celeba', 'pdbbind', 'bindingdb'],
                        help='name of dataset [%(default)d]')
    parser.add_argument('--root_path', type=str,
                        default='./')
    parser.add_argument('--v_size', type=int, default=30,
                        help='size of ground set [%(default)d]')
    parser.add_argument('--s_size', type=int, default=10,
                        help='size of subset [%(default)d]')
    parser.add_argument('--num_layers', type=int, default=2,
                        help='num layers [%(default)d]')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='batch size [%(default)d]')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='initial learning rate [%(default)g]')
    parser.add_argument("--weight_decay", type=float, default=1e-5,
                        help='weight decay rate [%(default)g]')
    parser.add_argument('--init', type=float, default=0.05,
                        help='unif init range (default if 0) [%(default)g]')
    parser.add_argument('--clip', type=float, default=10,
                        help='gradient clipping [%(default)g]')  # what is gradient clipping?
    parser.add_argument('--epochs', type=int, default=100,
                        help='max number of epochs [%(default)d]')
    parser.add_argument('--num_runs', type=int, default=1,
                        help='num random runs (not random if 1) '
                             '[%(default)d]')
    parser.add_argument('--num_bad_epochs', type=int, default=6,
                        help='num indulged bad epochs [%(default)d]')
    parser.add_argument('--num_workers', type=int, default=2,
                        help='num dataloader workers [%(default)d]')
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed [%(default)d]')
    parser.add_argument('--mode', type=str, default='implicit',
                        choices=['implicit', 'diffMF', 'ind', 'copula'],
                        help='name of the variant model [%(default)s]')
    parser.add_argument('--RNN_steps', type=int, default=1,
                        help='num of RNN steps [%(default)d], K in the paper')
    parser.add_argument('--num_samples', type=int, default=5,
                        help='num of Monte Carlo samples [%(default)d]')
    parser.add_argument('--derandomize', type=bool, default=False,
                        help='Flag to determine degree of randomization of the Monte-Carlo sampling [%(default)d]')
    parser.add_argument('--rank', type=int, default=5,
                        help='rank of the perturbation matrix [%(default)d]')
    parser.add_argument('--tau', type=float, default=0.1,
                        help='temperature of the relaxed multivariate bernoulli [%(default)g]')
    parser.add_argument('--neg_num', type=int, default=1,
                        help='num of the negative item [%(default)d]')
    parser.add_argument('--amazon_cat', type=str, default='toys',
                        choices=['toys', 'furniture', 'gear', 'carseats', 'bath', 'health', 'diaper', 'bedding',
                                 'safety', 'feeding', 'apparel', 'media'],
                        help='category of amazon baby registry dataset [%(default)d]')
    parser.add_argument('--IFT', type=bool, default=True,
                        help='Implicit differentiation flag [%(default)d]')
    parser.add_argument('--bwd_solver', type=str, default='normal_cg',
                        choices=['normal_cg', 'gmres'],
                        help='Backward solver choice to be used in backpropagation during implicit differentiation aka '
                             'solver of the linear system in implicit differentiation'
                             '[%(default)d]')
    parser.add_argument('--bwd_maxiter', type=int, default=20,
                        help='Number of fixed-point iterations [%(default)d]')
    parser.add_argument('--bwd_tol', type=float, default=1e-4,
                        help='Tolerance interval of fixed-point iterations [%(default)d]')
    parser.add_argument('--fwd_solver', type=str, default='fpi',
                        choices=['fpi', 'anderson'],
                        help='Forward solver choice to be used in forward pass during implicit differentiation '
                             '[%(default)d]')
    parser.add_argument('--fwd_maxiter', type=int, default=10,
                        help='Number of fixed-point iterations [%(default)d]')
    parser.add_argument('--fwd_tol', type=float, default=1e-7,
                        help='Tolerance interval of fixed-point iterations [%(default)d]')
    parser.add_argument('--is_verbose', type=bool, default=False,
                        help='Verbosity flag for JaxOPT fixed-point iterations [%(default)d]')
    args = parser.parse_args()
    return args


def set_params(args):
    config_name = args.data_name.upper() + '_CONFIG'
    config = getattr(config_file, config_name)
    print(f'getting config:{config}')
    if config:
        print(f'config set to {args.data_name.upper()}_CONFIG')
        args.v_size = config.get('v_size', args.v_size)
        args.s_size = config.get('s_size', args.s_size)
        args.batch_size = config.get('batch_size', args.batch_size)
    return args


if __name__ == "__main__":
    params = parse_arguments()
    if not params.manual_params:
        params = set_params(params)
    print("Current Arguments:", vars(params))
    data = get_data(params)
    CHECKPOINT_PATH = '../checkpoints/'
    batch_size = params.batch_size
    num_workers = params.num_workers


    def tensor_to_numpy(x):
        x = np.array(x, dtype=np.float32)
        return x


    train_loader, val_loader, test_loader = data.get_loaders(batch_size, num_workers, transform=tensor_to_numpy)
    # print(next(iter(train_loader))[0].shape)

    start_time = time.time()
    # Track memory usage before training
    process = psutil.Process(os.getpid())
    memory_before = process.memory_info().rss

    trainer = EquiVSetTrainer(params=params,
                              dim_feature=256,
                              optimizer_hparams={'lr': 0.0001},
                              logger_params={'base_log_dir': CHECKPOINT_PATH},
                              exmp_input=next(iter(train_loader)),
                              check_val_every_n_epoch=1,
                              seed=params.seed,
                              debug=False,
                              enable_progress_bar=False)

    metrics = trainer.train_model(train_loader,
                                  val_loader,
                                  test_loader=test_loader,
                                  num_epochs=params.epochs)
    print(metrics)
    # Track memory usage after training
    memory_after = process.memory_info().rss
    memory_used = memory_after - memory_before

    # Record end time
    end_time = time.time()
    elapsed_time = end_time - start_time

    # Create a dictionary of metrics
    metrics_dict = {
        'seed': params.seed,
        'data_name': params.data_name,
        'amazon_cat': params.amazon_cat,
        'mode': params.mode,
        'best_train_loss': metrics["best_train/loss"],
        'best_train_jaccard': metrics["best_train/jaccard"],
        'best_val_jaccard': metrics["best_val/jaccard"],
        'best_test_jaccard': metrics["best_test/jaccard"],
        'epoch_time': metrics["best_epoch_time"],
        'time': elapsed_time,
        'memory_used_MB': memory_used / (1024 ** 2),
        'train_loss': metrics["train/loss"],
        'train_jaccard': metrics["train/jaccard"],
        'val_jaccard': metrics["val/jaccard"],
    }
    # Convert dictionary to DataFrame
    df_metrics = pd.DataFrame([metrics_dict])

    # Filepath for the output CSV
    output_filepath = f"../history/metrics_{params.data_name}.csv"

    # Append metrics to CSV
    if os.path.exists(output_filepath):
        # If the file exists, load the existing DataFrame and concatenate
        output = pd.read_csv(output_filepath)
        output = pd.concat([output, df_metrics], ignore_index=True)
    else:
        # If the file does not exist, create a new DataFrame
        output = df_metrics

    # Save the updated DataFrame to CSV
    output.to_csv(output_filepath, index=False)

    # Print the head of the DataFrame
    print(output.head())

    print(f'Training Loss: {metrics["best_train/loss"]:.2f}')
    print(f'Training Jaccard index: {metrics["best_train/jaccard"]}')
    print(f'Validation Jaccard Index: {metrics["best_val/jaccard"]:.2f}')
    print(f'Test Jaccard Index: {metrics["best_test/jaccard"]:.2f}')
