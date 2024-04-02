# import torch
import datetime
# import torch.nn as nn
# import torch.nn.functional as F
import flax
import flax.linen as nn
import jax.numpy as jnp
from utils.flax_helper import FF, normal_cdf


def cross_entropy(q, S, neg_S):  # Eq. (5) in the paper
    loss = - jnp.sum((S * jnp.log(q + 1e-12) + (1 - S) * jnp.log(1 - q + 1e-12)) * neg_S, dim=-1)
    return loss.mean()


def fwd_solver(f, psi_init):
    psi_prev, psi = psi_init, f(psi_init)
    while jnp.linalg.norm(psi_prev - psi) > 1e-5:
        psi_prev, psi = psi, f(psi)
    return psi


class SetFunction(nn.Module):  # nn.Module is the base class for all NN modules. Any model should subclass this class.
    """
        Definition of the set function (F_θ) using a NN.
    """

    def __init__(self, params):
        super(SetFunction, self).__init__()
        self.params = params  # params = {v_size: 30,
        #                                 s_size: 10,
        #                                 num_layers: 2,
        #                                 batch_size: 4,
        #                                 lr: 0.0001,
        #                                 weight_decay: 1e-5,
        #                                 init: 0.05,
        #                                 clip: 10,
        #                                 epochs: 100,
        #                                 num_runs: 1,
        #                                 num_bad_epochs: 6,
        #                                 num_workers: 2
        #                                 }
        self.dim_feature = 256  # dimension of the NN layers

        self.init_layer = self.define_init_layer()  # custom init layers for different setups
        self.ff = FF(self.dim_feature, 500, 1, self.params.num_layers)  # forward fold

    def define_init_layer(self):
        """
        Returns the initial layer custom to different setups.
        :return: InitLayer
        """
        return nn.Dense(features=self.dim_feature)

    @staticmethod
    def MC_sampling(q, M):  # we should be able to get rid of this step altogether
        """
        Bernoulli sampling using q as parameters.
        Args:
            q: parameter of Bernoulli distribution (ψ in the paper)
            M: number of samples (m in the paper)

        Returns:
            Sampled subsets F(S+i), F(S)

        """
        bs, vs = q.shape
        q = q.reshape(bs, 1, 1, vs).expand(bs, M, vs, vs)  # is bs = 1?
        sample_matrix = jax.random.bernoulli(q)  # we should have torch uniform outside

        mask = jnp.cat([jnp.eye(vs, vs).unsqueeze(0) for _ in range(M)], dim=0).unsqueeze(0).to(q.device)
        # what does this line do?
        # after the first unsqueeze we have a 3D tensor with 1 channel, vs rows, and vs columns
        # after concat we have a 3D tensor with M channels, vs rows, and vs columns
        # after the second unsqueeze we have a (1, M, vs, vs) shaped tensor
        matrix_0 = sample_matrix * (1 - mask)  # element_wise multiplication
        matrix_1 = matrix_0 + mask
        return matrix_1, matrix_0  # F([x]_+i), F([x]_- i)

    def mean_field_iteration(self, V, subset_i, subset_not_i):  # ψ_i in the paper
        q = jax.nn.sigmoid((self.grad_F_S(V, subset_i, subset_not_i)).mean(1))
        return q

    def forward(self, V, S, neg_S, rec_net):  # return cross-entropy loss
        bs, vs = V.shape[:2]
        q = .5 * jnp.ones(bs, vs).to(V.device)  # ψ_0 <-- 0.5 * vector(1)

        # for i in range(self.params.RNN_steps):  # MFVI K times where K = RNN_steps
        #     sample_matrix_1, sample_matrix_0 = self.MC_sampling(q, self.params.num_samples)
        #     q = self.mean_field_iteration(V, sample_matrix_1, sample_matrix_0)  # ψ

        # there should be an alternative q definition here using IFT and AutoGrad
        # @partial(jax.custom_vjp, nondiff_argnums=(0, 1))
        # def fixed_point_layer(solver, f, params):
        #     psi_star = solver(lambda psi: f(params, psi), psi_init=jnp.zeros_like(x))
        #     return psi_star
        #
        # def fixed_point_layer_fwd(solver, f, params):
        #     psi_star = fixed_point_layer(solver, f, params)
        #     return psi_star, (params, psi_star)
        #
        # def fixed_point_layer_bwd(solver, f, res, psi_star_bar):
        #     params, x, psi_star = res
        #     _, vjp_a = jax.vjp(lambda params, x: f(params, psi_star), params)
        #     _, vjp_psi = jax.vjp(lambda psi: f(params, psi), psi_star)
        #     return vjp_a(solver(lambda u: vjp_psi(u)[0] + psi_star_bar, psi_init=jnp.zeros_like(psi_star)))
        #
        # fixed_point_layer.defvjp(fixed_point_layer_fwd, fixed_point_layer_bwd)
        #
        # theta = random.normal(random.PRNGKey(0), (ndim, ndim)) / jnp.sqrt(ndim)
        #
        # psi_star = fixed_point_layer(fwd_solver, f, theta)
        #
        # theta = theta - eta * jax.grad(lambda theta: fixed_point_layer(fwd_solver, f, theta, x).sum())(theta)

        loss = self.cross_entropy(q, S, neg_S)
        return loss

    def F_S(self, V, subset_mat, fpi=False):
        if fpi:
            # to fix point iteration (aka mean-field iteration)
            fea = self.init_layer(V).reshape(subset_mat.shape[0], 1, -1, self.dim_feature)
        else:
            # to encode variational dist
            fea = self.init_layer(V).reshape(subset_mat.shape[0], -1, self.dim_feature)
        fea = subset_mat @ fea
        fea = self.ff(fea)
        return fea

    def multilinear_relaxation(F_S, y):
        out = 0.0
        for i in range(2 ** len(y)):
            binary_vector = map(int, list(bin(i)[2:]))
            if len(binary_vector) < len(y):
                binary_vector = [0] * (len(y) - len(binary_vector)) + binary_vector
            x = dict(zip(y.iterkeys(), binary_vector))
            new_term = F_S(x)
            for key in y:
                if x[key] == 0:
                    new_term *= (1.0 - y[key])
                else:
                    new_term *= y[key]
            out += new_term
        return out

    def grad_F_S(self, V, subset_i, subset_not_i):
        F_1 = self.F_S(V, subset_i, fpi=True).squeeze(-1)
        F_0 = self.F_S(V, subset_not_i, fpi=True).squeeze(-1)
        return F_1 - F_0

    def hess_F_S(self, V, subset_ij, subset_i_not_j, subset_j_not_i, subset_not_ij):
        # one more squeeze might be necessary
        F_11 = self.F_S(V, subset_ij, fpi=True).squeeze(-1)
        F_10 = self.F_S(V, subset_i_not_j, fpi=True).squeeze(-1)
        F_01 = self.F_S(V, subset_j_not_i, fpi=True).squeeze(-1)
        F_00 = self.F_S(V, subset_not_ij, fpi=True).squeeze(-1)
        return F_11 - F_10 - F_01 + F_00

if __name__ == "__main__":
    params = {v_size: 30, s_size: 10, num_layers: 2, batch_size: 4, lr: 0.0001, weight_decay: 1e-5, init: 0.05,
              clip: 10, epochs: 100, num_runs: 1, num_bad_epochs: 6, num_workers: 2}
    mySet = SetFunction(params)

    subset_i, subset_not_i = mySet.MC_sampling(q, 500)

    



