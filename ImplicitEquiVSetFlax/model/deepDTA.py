# reference: https://github.com/mims-harvard/TDC/blob/main/examples/multi_pred/dti_dg/domainbed/networks.py
# import torch
# import torch.nn as nn
from jax import numpy as jnp
from flax import linen as nn
from typing import Dict
import torch.nn.functional as F
from torch.autograd import Variable


class Identity(nn.Module):
    """An identity layer"""

    @nn.compact
    def __call__(self, x):
        return x


class MLP(nn.Module):
    n_outputs: int
    hparams: Dict

    """Just an MLP"""

    def setup(self):
        self.input = nn.Dense(features=self.hparams['mlp_width'])
        self.dropout = nn.Dropout(self.hparams['mlp_dropout'])
        self.hiddens = [nn.Dense(features=self.hparams['mlp_width']) for _ in range(hparams['mlp_depth'] - 2)]
        self.output = nn.Dense(features=self.n_outputs)

    def __call__(self, x):
        x = self.input(x)
        x = self.dropout(x)
        x = nn.relu(x)
        for hidden in self.hiddens:
            x = hidden(x)
            x = self.dropout(x)
            x = nn.relu(x)
        x = self.output(x)
        return x


class CNN(nn.Module):
    encoding: str

    def setup(self):
        if self.encoding == 'drug':
            in_ch = [41] + [32, 64, 96]
            kernels = [4, 6, 8]
            layer_size = 3
            self.conv = [nn.Conv(features=in_ch[i + 1], kernel_size=kernels[i], padding='SAME') for i in
                         range(layer_size)]
            self.fc1 = nn.Dense(features=256)
        elif self.encoding == 'protein':
            in_ch = [20] + [32, 64, 96]
            kernels = [4, 8, 12]
            layer_size = 3
            self.conv = [nn.Conv(features=in_ch[i + 1], kernel_size=kernels[i]) for i in range(layer_size)]
            self.fc1 = nn.Dense(features=256)

    def _forward_features(self, x):
        for l in self.conv:
            # print(f"shape of l(v) is : {jnp.transpose(l(x), (0, 2, 1)).shape}")
            x = nn.relu(jnp.transpose(l(x), (0, 2, 1)))
        # print(f"shape of v is before max pool: {x.shape}")
        # Get the current size of the input
        input_size = x.shape[-1]
        output_size = 1

        # Calculate the kernel size and stride
        stride = input_size // output_size
        kernel_size = input_size - (output_size - 1) * stride
        x = nn.max_pool(x, window_shape=(kernel_size,), strides=(stride,), padding='VALID')
        # print(f"shape of v is after max pool: {x.shape}")
        return x

    def __call__(self, v):
        # print(f"shape of v is: {v.shape}")  # (1200, 41, 100)
        v = self._forward_features(v)
        # print(f"shape of v is after max pool: {v.shape}")
        v = v.reshape((v.shape[0], -1))
        # print(f"shape of v is after reshape: {v.shape}")
        v = self.fc1(v)
        return v


class DeepDTA_Encoder(nn.Module):
    def setup(self):
        # super(DeepDTA_Encoder, self).__init__()
        # self.input_dim_drug = 256
        # self.input_dim_protein = 256
        self.model_drug = CNN('drug')
        self.model_protein = CNN('protein')
        self.predictor = nn.Dense(features=256)

    def __call__(self, V):
        v_D, v_P = V
        # each encoding
        v_D = self.model_drug(v_D)
        v_P = self.model_protein(v_P)

        # print(f"shape of v_D is {v_D.shape}")
        # print(f"shape of v_P is {v_P.shape}")

        # concatenate and output feature
        v_f = jnp.concatenate((v_D, v_P), axis=1)
        # print(f"shape of v_f after concat is {v_f.shape}")
        v_f = self.predictor(v_f)
        # print(f"shape of v_f after predictor is {v_f.shape}")
        return v_f
