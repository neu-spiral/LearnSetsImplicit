import sys

# setting path
sys.path.append('../ImplicitEquiVSetFlax')
import datetime
import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from functools import partial
from jaxopt import AndersonAcceleration, FixedPointIteration
from jaxopt.linear_solve import solve_gmres, solve_normal_cg
from model.acnn import ACNN
from utils.config import ACNN_CONFIG
from model.celebaCNN import CelebaCNN
from model.deepDTA import DeepDTA_Encoder
from typing import Callable
from utils.flax_helper import FF, normal_cdf
from utils.implicit import SigmoidImplicitLayer


class MFVI(nn.Module):
    params: dict
    dim_feature: int = 256

    # noinspection PyMethodMayBeStatic
    def MC_sampling(self, q, M, derandomize=False):
        """
        Bernoulli sampling using q as parameters.
        Args:
            q: parameter of Bernoulli distribution (ψ in the paper)
            M: number of samples (m in the paper)

        Returns:
            Sampled subsets F(S+i), F(S)
            :param derandomize:

        """
        bs, vs = q.shape
        if derandomize:
            # same sample is used for each coordinate
            q = jnp.broadcast_to(q.reshape(bs, 1, vs), (bs, M, vs))
        else:
            # a new sample is generated for each coordinate
            q = jnp.broadcast_to(q.reshape(bs, 1, 1, vs), (bs, M, vs, vs))
        q = jax.device_put(q)
        sample_matrix = jax.random.bernoulli(jax.random.PRNGKey(self.params.seed), q)
        if derandomize:
            sample_matrix = jnp.broadcast_to(sample_matrix.reshape(bs, M, 1, vs), (bs, M, vs, vs))

        mask = jnp.expand_dims(jnp.concatenate([jnp.expand_dims(jnp.eye(vs, vs), axis=0) for _ in range(M)], axis=0),
                               axis=0)

        matrix_0 = sample_matrix * (1 - mask)  # element_wise multiplication
        matrix_1 = matrix_0 + mask
        return matrix_1, matrix_0, sample_matrix  # F([x]_+i), F([x]_- i)

    # @property
    def define_init_layer(self):
        """
        Returns the initial layer custom to different setups.
        :return: InitLayer
        """
        data_name = self.params.data_name
        if data_name == 'celeba':
            return CelebaCNN()
        elif data_name == 'bindingdb':
            return DeepDTA_Encoder()
        # elif data_name == 'pdbbind':
        #     return ACNN(hidden_sizes=ACNN_CONFIG['hidden_sizes'],
        #                 weight_init_stddevs=ACNN_CONFIG['weight_init_stddevs'],
        #                 dropouts=ACNN_CONFIG['dropouts'],
        #                 features_to_use=ACNN_CONFIG['atomic_numbers_considered'],
        #                 radial=ACNN_CONFIG['radial'])
        return nn.Dense(features=self.dim_feature)

    @nn.compact
    def __call__(self, q, V):  # ψ_i in the paper
        # returns S+i , S
        subset_i, subset_not_i, _ = self.MC_sampling(q, self.params.num_samples, derandomize=self.params.derandomize)
        init_layer = self.define_init_layer()
        ff = FF(self.dim_feature, 500, 1, self.params.num_layers)
        # print(type(V))
        # print(V)
        # print(init_layer(V).shape)  # this shape should be (1024, 256) instead of (1024, 64, 64, 256)
        fea_1 = init_layer(V).reshape(subset_i.shape[0], 1, -1, self.dim_feature)
        fea_1 = subset_i @ fea_1
        fea_1 = ff(fea_1).squeeze(-1)

        fea_0 = init_layer(V).reshape(subset_not_i.shape[0], 1, -1, self.dim_feature)
        fea_0 = subset_not_i @ fea_0
        fea_0 = ff(fea_0).squeeze(-1)
        f_max = jnp.absolute(jnp.maximum(jnp.max(fea_0), jnp.max(fea_1)))

        # self.params.M = jnp.where(f_max > self.params.M, f_max, self.params.M)

        grad = (fea_1 - fea_0).mean(1)
        norm = jnp.linalg.norm(grad, ord=self.params.norm)
        # jax.debug.print("norm: {}", norm)
        # if norm > self.params.M:
        #     self.params.M = norm
        # self.params.M = jnp.where(norm > self.params.M, norm, self.params.M)
        # grad = jnp.where(l2_norm > 2/self.params.v_size, (2 / (self.params.v_size * l2_norm)) * grad, grad)

        q = jax.nn.sigmoid((2 / (self.params.v_size * norm)) * grad)
        # q = jax.nn.sigmoid(grad)
        return q


class CrossEntropy(nn.Module):
    @nn.compact
    def __call__(self, q, S, neg_S):  # Eq. (5) in the paper
        loss = - jnp.sum((S * jnp.log(q + 1e-12) + (1 - S) * jnp.log(1 - q + 1e-12)) * neg_S, axis=-1)
        return loss.mean()


# noinspection PyAttributeOutsideInit
class SetFunction(nn.Module):
    """
        Definition of the set function (F_θ) using a NN.
    """
    params: dict
    dim_feature: int = 256

    def setup(self):
        self.mfvi = MFVI(self.params)
        self.fixed_point_layer = self.define_fixed_point_layer(self.mfvi)
        self.cross_entropy = CrossEntropy()

    def define_fixed_point_layer(self, mfvi):
        """

        :return:
        """
        if self.params.bwd_solver == 'normal_cg':
            implicit_solver = partial(solve_normal_cg, tol=self.params.bwd_tol, maxiter=self.params.bwd_maxiter)
        elif self.params.bwd_solver == "gmres":
            implicit_solver = partial(solve_gmres, tol=self.params.bwd_tol, maxiter=self.params.bwd_maxiter)
        if self.params.fwd_solver == 'fpi':
            fixed_point_solver = partial(FixedPointIteration,
                                         maxiter=self.params.fwd_maxiter,
                                         tol=self.params.fwd_tol, implicit_diff=self.params.IFT,
                                         implicit_diff_solve=implicit_solver,
                                         verbose=self.params.is_verbose)
        elif self.params.fwd_solver == 'anderson':
            fixed_point_solver = partial(AndersonAcceleration,
                                         history_size=self.params.anderson_hist_size,
                                         ridge=self.params.anderson_ridge,
                                         maxiter=self.params.fwd_maxiter,
                                         tol=self.params.fwd_tol, implicit_diff=self.params.IFT,
                                         implicit_diff_solve=implicit_solver,
                                         verbose=self.params.is_verbose)
        return SigmoidImplicitLayer(mfvi=mfvi, fixed_point_solver=fixed_point_solver)

    def __call__(self, V, S, neg_S, rec_net=None, **kwargs):
        """"returns cross-entropy loss."""
        if self.params.mode == 'implicit':
            # print(V)
            if self.params.data_name == 'bindingdb':
                bs = self.params.batch_size
                vs = self.params.v_size
            else:
                bs, vs = V.shape[:2]
                if self.params.data_name == 'celeba':
                    bs = int(bs / 8)
                    vs = self.params.v_size
            q = .5 * jnp.ones((bs, vs))  # ψ_0 <-- 0.5 * vector(1)
        else:
            q = rec_net.get_vardist(V, S.shape[0])
        # V = jnp.array(V)
        # print("Shape of V:", jnp.shape(V))
        q = self.fixed_point_layer(q, V)

        loss = self.cross_entropy(q, S, neg_S)
        return loss


# noinspection PyAttributeOutsideInit
class RecNet(nn.Module):  # this is only used for 'ind' or 'copula' modes
    params: dict
    dim_feature: int = 256

    def setup(self):
        self.init_layer = self.define_init_layer()
        self.ff = FF(self.dim_feature, 500, 500, self.params.num_layers - 1 if self.params.num_layers > 0 else 0)
        self.h_to_mu = nn.Dense(features=1)
        if self.params.mode == 'copula':
            self.h_to_std = nn.Dense(features=1)
            self.h_to_U = nn.ModuleList([nn.Dense(features=1) for _ in range(self.params.rank)])

    def define_init_layer(self):
        data_name = self.params.data_name
        if data_name == 'celeba':
            return CelebaCNN()
        elif data_name == 'gaussian':
            return nn.Dense(features=self.dim_feature)
        elif data_name == 'amazon':
            return nn.Dense(features=self.dim_feature)
        elif data_name == 'moons':
            return nn.Dense(features=self.dim_feature)
        # elif data_name == 'pdbbind':
        #     return ACNN(hidden_sizes=ACNN_CONFIG['hidden_sizes'],
        #                 weight_init_stddevs=ACNN_CONFIG['weight_init_stddevs'],
        #                 dropouts=ACNN_CONFIG['dropouts'],
        #                 features_to_use=ACNN_CONFIG['atomic_numbers_considered'],
        #                 radial=ACNN_CONFIG['radial'])
        elif data_name == 'bindingdb':
            return DeepDTA_Encoder()
        else:
            raise ValueError("invalid dataset...")

    def encode(self, V, bs):
        """

        Args:
            V: the ground set. [batch_size, v_size, fea_dim]
            bs: batch_size

        Returns:
            ber: predicted probabilities.     [batch_size, v_size]
            std: the diagonal matrix D        [batch_size, v_size]
            u_perturbation:  the low rank perturbation matrix         [batch_size, v_size, rank]

        """
        fea = self.init_layer(V).reshape(bs, -1, self.dim_feature)
        h = nn.relu(self.ff(fea))
        ber = jax.nn.sigmoid(self.h_to_mu(h)).squeeze(-1)  # [batch_size, v_size]

        if self.params.mode == 'copula':
            std = nn.softplus(self.h_to_std(h)).squeeze(-1)  # [batch_size, v_size]
            rs = []
            for i in range(self.params.rank):
                rs.append(nn.tanh(self.h_to_U[i](h)))
            u_perturbation = jnp.concatenateat(rs, axis=-1)  # [batch_size, v_size, rank]

            return ber, std, u_perturbation
        return ber, None, None

    def MC_sampling(self, ber, std, u_pert, M):
        """
        Sampling using CopulaBernoulli

        Args:
            ber: location parameter (0, 1)               [batch_size, v_size]
            std: standard deviation (0, +infinity)      [batch_size, v_size]
            u_pert: lower rank perturbation (-1, 1)     [batch_size, v_size, rank]
            M: number of MC approximation

        Returns:
            Sampled subsets
        """
        bs, vs = ber.shape

        if self.params.mode == 'copula':
            eps = jax.random.normal(jax.random.PRNGKey(self.params.seed), (bs, M, vs))
            eps_corr = jax.random.normal(jax.random.PRNGKey(self.params.seed), (bs, M, self.params.rank, 1))
            g = eps * jnp.expand_dims(std, axis=1) + jnp.matmul(jnp.expand_dims(u_pert, axis=1), eps_corr).squeeze(-1)
            u = normal_cdf(g, 0, 1)
        else:
            # mode == 'ind'
            u = jax.random.uniform(jax.random.PRNGKey(self.params.seed), (bs, M, vs))

        ber = jnp.expand_dims(ber, axis=1)
        l = jnp.log(ber + 1e-12) - jnp.log(1 - ber + 1e-12) + jnp.log(u + 1e-12) - jnp.log(1 - u + 1e-12)

        prob = jax.nn.sigmoid(l / self.params.tau)
        r = jax.random.bernoulli(jax.random.PRNGKey(self.params.seed), prob)  # binary vector
        s = prob + (r - prob)  # straight through estimator
        return s

    def cal_elbo(self, V, sample_mat, set_func, q):
        # f_mt = set_func.F_S(V, sample_mat).squeeze(-1).mean(-1)
        fea = self.init_layer(V).reshape(sample_mat.shape[0], -1, self.dim_feature)
        fea = sample_mat @ fea
        fea = self.ff(fea).squeeze(-1).mean(-1)
        entropy = - jnp.sum(q * jnp.log(q + 1e-12) + (1 - q) * jnp.log(1 - q + 1e-12), axis=-1)
        elbo = f_mt + entropy
        return elbo.mean()

    def __call__(self, V, set_func, bs):  # return negative ELBO
        ber, std, u_perturbation = self.encode(V, bs)
        sample_mat = self.MC_sampling(ber, std, u_perturbation, self.params.num_samples)
        elbo = self.cal_elbo(V, sample_mat, set_func, ber)
        return -elbo

    def get_vardist(self, V, bs):
        fea = self.init_layer(V).reshape(bs, -1, self.dim_feature)
        h = nn.relu(self.ff(fea))
        ber = jax.nn.sigmoid(self.h_to_mu(h)).squeeze(-1)
        return ber


if __name__ == "__main__":
    params = {'v_size': 30, 's_size': 10, 'num_layers': 2, 'batch_size': 4, 'lr': 0.0001, 'weight_decay': 1e-5,
              'init': 0.05, 'clip': 10, 'epochs': 100, 'num_runs': 1, 'num_bad_epochs': 6, 'num_workers': 2,
              'RNN_steps': 1, 'num_samples': 5}

    rng = jax.random.PRNGKey(42)
    rng, V_inp_rng, S_inp_rng, neg_S_inp_rng, rec_net_inp_rng, init_rng = jax.random.split(rng, 6)
    V_inp = jax.random.normal(V_inp_rng, (4, 100, 2))  # Batch size 4, input size (V) 100, dim_input 2
    S_inp = jax.random.normal(S_inp_rng, (4, 100))  # Batch size 4, input size (V) 100
    neg_S_inp = jax.random.normal(neg_S_inp_rng, (4, 100))  # Batch size 4, input size (V) 100
    rec_net_inp = jax.random.normal(rec_net_inp_rng, (4, 1))  # Batch size 4, input size (V) 100
    # MOONS_CONFIG = {'data_name': 'moons', 'v_size': 100, 's_size': 10, 'batch_size': 128}
    # x = random.normal(key1, (10,))  # Dummy input data
    # params = model.init(key2, x)  # Initialization call
    # jax.tree_util.tree_map(lambda x: x.shape, params)  # Checking output shapes
    mySetModel = SetFunction(params=params)
    print(mySetModel)
    new_params = mySetModel.init(init_rng, V_inp, S_inp, neg_S_inp)
    print(new_params)

    # subset_i, subset_not_i = mySet.MC_sampling(q, 500)
