import dgl
import math
import jax
import flax
import pickle
import flax.linen as nn
import jax.numpy as jnp
from typing import Callable


def set_value_according_index(tensor, idx, value):
    mask_val = torch.ones(idx.shape).to(tensor.device) * value
    tensor.scatter_(1, idx, mask_val)  # fill the values along dimension 1
    return tensor


def normal_cdf(value, loc, scale):
    return 0.5 * (1 + torch.erf((value - loc) / (scale * math.sqrt(2))))


def get_init_function(init_value):
    def init_function(m):
        if init_value > 0.:
            if hasattr(m, 'weight'):
                m.weight.data.uniform_(-init_value, init_value)
            if hasattr(m, 'bias'):
                m.bias.data.fill_(0.)

    return init_function


class SigmoidFixedPointLayer(nn.Module):
    set_func: Callable
    samp_func: Callable
    num_samples: int = 5
    tol: float = 1e-3
    max_iter: int = 100

    def __call__(self, x, V, **kwargs):
        # initialize output q to be 0.5
        # bs, vs = x.shape
        # q = .5 * jnp.ones((bs, vs))
        q = x
        iterations = 0
        errs = []
        last_err = float('inf')

        subset_i, subset_not_i, _ = self.samp_func(q, self.num_samples)
        key = jax.random.key(42)
        V_dummy = jnp.ones(shape=V.shape)
        s1_dummy = jnp.ones(shape=subset_i.shape)
        s0_dummy = jnp.ones(shape=subset_not_i.shape)
        init_params = self.set_func.init(key, V_dummy, s1_dummy, s0_dummy, method='grad_F_S')

        # iterate until convergence
        while iterations < self.max_iter:
            subset_i, subset_not_i, _ = self.samp_func(q, self.num_samples)
            grad_set_func = self.set_func.apply(init_params, V, subset_i, subset_not_i, method='grad_F_S')
            q_next = jax.nn.sigmoid(grad_set_func)
            err = jnp.linalg.norm(q - q_next)
            errs.append(err)
            q = q_next
            iterations += 1
            if err < self.tol or last_err < err:
                break
            last_err = err

        return q, iterations, err, errs


class SigmoidImplicitLayer(nn.Module):
    tol: float = 1e-3
    max_iter: int = 100

    def __init__(self, out_features, tol=1e-4, max_iter=50):
        super().__init__()
        self.linear = nn.Linear(out_features, out_features, bias=False)
        self.tol = tol
        self.max_iter = max_iter

    def __call__(self, x):
        # Run Newton's method outside of the autograd framework
        with torch.no_grad():
            z = torch.tanh(x)
            self.iterations = 0
            while self.iterations < self.max_iter:
                z_linear = self.linear(z) + x
                g = z - torch.tanh(z_linear)
                self.err = torch.norm(g)
                if self.err < self.tol:
                    break

                # newton step
                J = torch.eye(z.shape[1])[None, :, :] - (1 / torch.cosh(z_linear) ** 2)[:, :,
                                                        None] * self.linear.weight[None, :, :]
                z = z - torch.solve(g[:, :, None], J)[0][:, :, 0]
                self.iterations += 1

        # reengage autograd and add the gradient hook
        z = torch.tanh(self.linear(z) + x)
        z.register_hook(lambda grad: torch.solve(grad[:, :, None], J.transpose(1, 2))[0][:, :, 0])
        return z


# noinspection PyAttributeOutsideInit
# feed forward
class FF(nn.Module):
    dim_input: int
    dim_hidden: int
    dim_output: int
    num_layers: int
    activation: str = 'relu'
    dropout_rate: float = 0.0
    layer_norm: bool = False
    residual_connection: bool = False

    @nn.compact
    def __call__(self, x, **kwargs):
        if not self.residual_connection:
            for l in range(self.num_layers):
                if self.layer_norm:
                    x = nn.LayerNorm()(x)
                x = nn.Dense(features=self.dim_hidden)(x)
                x = {'tanh': nn.tanh, 'relu': nn.relu}[self.activation](x)
                if self.dropout_rate > 0.0:
                    x = nn.Dropout(self.dropout_rate)(x)
            x = nn.Dense(features=self.dim_output)(x)
        else:
            for l in range(self.num_layers):
                if self.layer_norm:
                    x += nn.LayerNorm()(x)
                x += nn.Dense(features=self.dim_hidden)(x)
                x += {'tanh': nn.tanh, 'relu': nn.relu}[self.activation](x)
                if self.dropout_rate > 0.0:
                    x += nn.Dropout(self.dropout_rate)(x)
            x += nn.Dense(features=self.dim_output)(x)
        return x

    # def setup(self):
    #     assert num_layers >= 0  # 0 = Linear
    #     if num_layers > 0:
    #         assert dim_hidden > 0
    #     if residual_connection:
    #         assert dim_hidden == dim_input
    #
    #     self.residual_connection = residual_connection
    #     self.stack = []  # this will just be a list probably
    #     for l in range(num_layers):
    #         layer = []
    #
    #         if layer_norm:
    #             layer.append(nn.LayerNorm())
    #
    #         layer.append(nn.Dense(features=dim_hidden))
    #         layer.append({'tanh': nn.tanh, 'relu': nn.relu}[activation])
    #         # layer.append(nn.relu if activation == 'relu' else nn.tanh)
    #
    #         if dropout_rate > 0.0:
    #             layer.append(nn.Dropout(dropout_rate))
    #
    #         self.stack.append(nn.Sequential(layer))
    #
    #     self.out = nn.Dense(features=dim_output)
    #
    # def __call__(self, x):
    #     for layer in self.stack:
    #         x = x + layer(x) if self.residual_connection else layer(x)
    #     return self.out(x)


def read_from_pickle(filename):
    filename = filename + '.pickle'
    with open(filename, 'rb') as f:
        x = pickle.load(f)
        return x


def find_not_in_set(U, S):
    Ind = jnp.ones(U.shape[0], dtype=bool)
    Ind[S] = False
    return U[Ind]
