import dgl
import math
import flax
import pickle
import flax.linen as nn


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


# no need for flax
def move_to_device(obj, device):
    if torch.is_tensor(obj):
        return obj.to(device)
    elif isinstance(obj, dgl.DGLGraph):
        return obj.to(device)
    elif isinstance(obj, dict):
        res = {}
        for k, v in obj.items():
            res[k] = move_to_device(v, device)
        return res
    elif isinstance(obj, list):
        res = []
        for v in obj:
            res.append(move_to_device(v, device))
        return res
    elif isinstance(obj, tuple):
        res = ()
        for v in obj:
            res += (move_to_device(v, device),)
        return res
    else:
        raise TypeError("Invalid type for move_to_device")


# noinspection PyAttributeOutsideInit
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
