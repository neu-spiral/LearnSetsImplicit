import sys

# setting path
sys.path.append('../MyEquiVSetFlax')
import jax
import datetime
import os
import itertools
import flax
import flax.linen as nn
import jax.numpy as jnp
import numpy as np
from utils.flax_helper import FF, SigmoidFixedPointLayer, normal_cdf
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('Agg')


def cross_entropy(q, S, neg_S):  # Eq. (5) in the paper
    # jax.debug.print("S is {S}\n", S=S)
    # jax.debug.print("shape of S is {S.shape}\n", S=S)
    # jax.debug.print("q is {q}\n", q=q)
    # jax.debug.print("shape of q is {q.shape}\n", q=q)
    # jax.debug.print("neg_S is {neg_S}\n", neg_S=neg_S)
    # jax.debug.print("shape of neg_S is {neg_S.shape}\n", neg_S=neg_S)
    loss = - jnp.sum((S * jnp.log(q + 1e-12) + (1 - S) * jnp.log(1 - q + 1e-12)) * neg_S, axis=-1)
    return loss.mean()


def fwd_solver(f, psi_init):
    psi_prev, psi = psi_init, f(psi_init)
    while jnp.linalg.norm(psi_prev - psi) > 1e-5:
        psi_prev, psi = psi, f(psi)
    return psi


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
    # uncomment below for the fully randomized version
    # q = jnp.broadcast_to(q.reshape(bs, 1, 1, vs), (bs, M, vs, vs))  # a new sample is generated for each coordinate

    # uncomment below for the de-randomized version
    q = jnp.broadcast_to(q.reshape(bs, 1, vs), (bs, M, vs))  # same sample is used for each coordinate
    # bs = the batch size
    q = jax.device_put(q)  # not sure if we need this
    sample_matrix = jax.random.bernoulli(jax.random.PRNGKey(758493), q)

    # uncomment below for the fully randomized version
    sample_matrix = jnp.broadcast_to(sample_matrix.reshape(bs, M, 1, vs), (bs, M, vs, vs))

    mask = jnp.expand_dims(jnp.concatenate([jnp.expand_dims(jnp.eye(vs, vs), axis=0) for _ in range(M)], axis=0),
                           axis=0)
    # what does this line do?
    # after the first unsqueeze we have a 3D tensor with 1 channel, vs rows, and vs columns
    # after concat we have a 3D tensor with M channels, vs rows, and vs columns
    # after the second unsqueeze we have a (1, M, vs, vs) shaped tensor
    # q and sample_matrix are (bs, M, vs, vs) shaped
    # mask is (1, M, vs, vs) shaped
    matrix_0 = sample_matrix * (1 - mask)  # element_wise multiplication
    matrix_1 = matrix_0 + mask
    # print(matrix_0)
    # print(matrix_1)
    # print(sample_matrix)
    return matrix_1, matrix_0, sample_matrix  # F([x]_+i), F([x]_- i)


# noinspection PyAttributeOutsideInit
class SetFunction(nn.Module):  # nn.Module is the base class for all NN modules. Any model should subclass this class.
    """
        Definition of the set function (F_θ) using a NN.
    """
    params: dict
    dim_feature: int = 256

    # For instance, if we define more functions on a module besides __call__ and want to reuse some modules, it is
    # recommended to use the setup version.
    def setup(self):
        self.init_layer = self.define_init_layer()
        self.ff = FF(self.dim_feature, 500, 1, self.params['num_layers'])
        self.fixed_point_layer = SigmoidFixedPointLayer(self.grad_F_S, MC_sampling,
                                                        num_samples=self.params['num_samples'])

    def define_init_layer(self):
        """
        Returns the initial layer custom to different setups.
        :return: InitLayer
        """
        return nn.Dense(features=self.dim_feature)

    def mean_field_iteration(self, V, subset_i, subset_not_i):  # ψ_i in the paper, I can call this as a layer
        # print((self.grad_F_S(V, subset_i, subset_not_i)).mean(1).shape)
        q = jax.nn.sigmoid(self.grad_F_S(V, subset_i, subset_not_i))
        return q

    def __call__(self, V, S, neg_S, **kwargs):
        """"returns cross-entropy loss."""
        bs, vs = V.shape[:2]
        q = .5 * jnp.ones((bs, vs))  # ψ_0 <-- 0.5 * vector(1)
        # q = jax.random.uniform(jax.random.PRNGKey(758493), shape=(bs, vs))

        for i in range(self.params['RNN_steps']):  # MFVI K times where K = RNN_steps
            sample_matrix_1, sample_matrix_0, _ = MC_sampling(q, self.params['num_samples'])  # returns S+i , S
            q = self.mean_field_iteration(V, sample_matrix_1, sample_matrix_0)  # ψ, ∇f^{Fθ}(ψ), Eq. 11

        # q = SigmoidFixedPointLayer(self.grad_F_S, MC_sampling, num_samples=self.params['num_samples'])(q)

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

        loss = cross_entropy(q, S, neg_S)
        # jax.debug.print("loss is {loss}\n", loss=loss)
        return loss

    def F_S(self, V, subset_mat, fpi=False):
        # print(self.init_layer(V).type)
        # print(self.init_layer(V).shape)
        if fpi:
            # to fix point iteration (aka mean-field iteration)
            fea = self.init_layer(V).reshape(subset_mat.shape[0], 1, -1, self.dim_feature)
        else:
            # to encode variational dist
            fea = self.init_layer(V).reshape(subset_mat.shape[0], -1, self.dim_feature)
        # print(subset_mat.shape)  # (bs, M, vs, vs)
        # print(fea.shape)  # (bs, 1, vs, dim_feature)
        fea = subset_mat @ fea
        # print(fea.shape)
        fea = self.ff(fea)  # goes through FF block
        # self.ff.apply(params, fea)
        # print(fea.shape)
        return fea

    def get_powerset(self, V):
        bs, vs = V.shape[:2]
        powerset = jnp.array(list(itertools.product([0, 1], repeat=vs)))
        powerset = jnp.concatenate([jnp.expand_dims(powerset, axis=0) for _ in range(bs)], axis=0)
        return powerset

    def multilinear_relaxation(self, V, powerset, q):
        bs, subsets, vs = powerset.shape
        expanded_powerset = jnp.broadcast_to(powerset.reshape(bs, subsets, 1, vs), (bs, subsets, vs, vs))
        fea = self.F_S(V, expanded_powerset, fpi=True).squeeze(-1)
        q = jnp.broadcast_to(q.reshape(bs, 1, vs), (bs, subsets, vs))
        # print(fea.shape)
        # print(q)
        probs = ((1 - powerset) + powerset * q) * (powerset + (1 - powerset) * (1 - q))
        probs = jnp.prod(probs, axis=-1)
        # print(probs.shape)
        # print(fea.mean(-1).shape)
        return jnp.sum(fea.mean(-1) * probs, axis=-1)

    def grad_F_S(self, V, subset_i, subset_not_i):
        F_1 = self.F_S(V, subset_i, fpi=True).squeeze(-1)
        F_0 = self.F_S(V, subset_not_i, fpi=True).squeeze(-1)
        return (F_1 - F_0).mean(1)

    def estimate_grad_F_S(self, V, subset_mat, delta):
        bs, M, vs = subset_mat.shape[:-1]
        mask = jnp.expand_dims(jnp.concatenate([jnp.expand_dims(jnp.eye(vs, vs), axis=0) for _ in range(M)], axis=0),
                           axis=0)
        F_delta = self.F_S(V, subset_mat + mask * delta, fpi=True).squeeze(-1)
        F = self.F_S(V, subset_mat, fpi=True).squeeze(-1)
        return (F_delta - F).mean(1) / delta

    # def get_q_i(self, q):  # we should be able to get rid of this step altogether
    #     """
    #     Args:
    #         q: parameter of Bernoulli distribution (ψ in the paper)
    #
    #     Returns:
    #         Sampled subsets F(S+i), F(S)
    #
    #     """
    #     bs, vs = q.shape
    #     subsets =  2 ** vs
    #     q = jnp.broadcast_to(q.reshape(bs, 1, 1, vs), (bs, subsets, vs, vs))
    #     mask = jnp.expand_dims(jnp.concatenate([jnp.expand_dims(jnp.eye(vs, vs), axis=0) for _ in range(subsets)], axis=0),
    #                            axis=0)
    #
    #     q_not_i = q * (1 - mask)  # element_wise multiplication
    #     q_i = q_not_i + mask
    #     # print(matrix_0)
    #     # print(matrix_1)
    #     return q_i, q_not_i  # F([x]_+i), F([x]_- i)
    #
    # def true_grad(self, V, q_i, q_not_i):
    #     powerset = self.get_powerset(V)
    #     F_1 = self.multilinear_relaxation(V, powerset, q_i)
    #     F_0 = self.multilinear_relaxation(V, powerset, q_not_i)
    #     return F_1 - F_0

    def hess_F_S(self, V, subset_ij, subset_i_not_j, subset_j_not_i, subset_not_ij):
        # one more squeeze might be necessary
        F_11 = self.F_S(V, subset_ij, fpi=True).squeeze(-1)
        F_10 = self.F_S(V, subset_i_not_j, fpi=True).squeeze(-1)
        F_01 = self.F_S(V, subset_j_not_i, fpi=True).squeeze(-1)
        F_00 = self.F_S(V, subset_not_ij, fpi=True).squeeze(-1)
        return F_11 - F_10 - F_01 + F_00


class RecNet(nn.Module):  # this is only used for 'ind' or 'copula' modes
    def __init__(self, params):
        super(RecNet, self).__init__()
        self.params = params
        self.dim_feature = 256
        num_layers = self.params.num_layers

        self.init_layer = self.define_init_layer()
        self.ff = FF(self.dim_feature, 500, 500, num_layers - 1 if num_layers > 0 else 0)
        self.h_to_mu = nn.Linear(500, 1)
        if self.params.mode == 'copula':
            self.h_to_std = nn.Linear(500, 1)
            self.h_to_U = nn.ModuleList([nn.Linear(500, 1) for i in range(self.params.rank)])

    def define_init_layer(self):
        data_name = self.params.data_name
        if data_name == 'moons':
            return nn.Linear(2, self.dim_feature)
        elif data_name == 'gaussian':
            return nn.Linear(2, self.dim_feature)
        elif data_name == 'amazon':
            return nn.Linear(768, self.dim_feature)
        elif data_name == 'celeba':
            return celebaCNN()
        elif data_name == 'pdbbind':
            return ACNN(hidden_sizes=ACNN_CONFIG['hidden_sizes'],
                        weight_init_stddevs=ACNN_CONFIG['weight_init_stddevs'],
                        dropouts=ACNN_CONFIG['dropouts'],
                        features_to_use=ACNN_CONFIG['atomic_numbers_considered'],
                        radial=ACNN_CONFIG['radial'])
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
        h = torch.relu(self.ff(fea))
        ber = torch.sigmoid(self.h_to_mu(h)).squeeze(-1)  # [batch_size, v_size]

        if self.params.mode == 'copula':
            std = F.softplus(self.h_to_std(h)).squeeze(-1)  # [batch_size, v_size]
            rs = []
            for i in range(self.params.rank):
                rs.append(torch.tanh(self.h_to_U[i](h)))
            u_perturbation = torch.cat(rs, -1)  # [batch_size, v_size, rank]

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
            eps = torch.randn((bs, M, vs)).to(ber.device)
            eps_corr = torch.randn((bs, M, self.params.rank, 1)).to(ber.device)
            g = eps * std.unsqueeze(1) + torch.matmul(u_pert.unsqueeze(1), eps_corr).squeeze(-1)
            u = normal_cdf(g, 0, 1)
        else:
            # mode == 'ind'
            u = torch.rand((bs, M, vs)).to(ber.device)

        ber = ber.unsqueeze(1)
        l = torch.log(ber + 1e-12) - torch.log(1 - ber + 1e-12) + \
            torch.log(u + 1e-12) - torch.log(1 - u + 1e-12)

        prob = torch.sigmoid(l / self.params.tau)
        r = torch.bernoulli(prob)  # binary vector
        s = prob + (r - prob).detach()  # straight through estimator
        return s

    def cal_elbo(self, V, sample_mat, set_func, q):
        f_mt = set_func.F_S(V, sample_mat).squeeze(-1).mean(-1)
        entropy = - torch.sum(q * torch.log(q + 1e-12) + (1 - q) * torch.log(1 - q + 1e-12), dim=-1)
        elbo = f_mt + entropy
        return elbo.mean()

    def forward(self, V, set_func, bs):  # return negative ELBO
        ber, std, u_perturbation = self.encode(V, bs)
        sample_mat = self.MC_sampling(ber, std, u_perturbation, self.params.num_samples)
        elbo = self.cal_elbo(V, sample_mat, set_func, ber)
        return -elbo

    def get_vardist(self, V, bs):
        fea = self.init_layer(V).reshape(bs, -1, self.dim_feature)
        h = torch.relu(self.ff(fea))
        ber = torch.sigmoid(self.h_to_mu(h)).squeeze(-1)
        return ber


if __name__ == "__main__":
    params = {'v_size': 30, 's_size': 10, 'num_layers': 2, 'batch_size': 1, 'lr': 0.0001, 'weight_decay': 1e-5,
              'init': 0.05, 'clip': 10, 'epochs': 100, 'num_runs': 1, 'num_bad_epochs': 6, 'num_workers': 2,
              'RNN_steps': 1, 'num_samples': 100}

    rng = jax.random.PRNGKey(42)
    rng, V_inp_rng, S_inp_rng, neg_S_inp_rng, rec_net_inp_rng, init_rng = jax.random.split(rng, 6)
    V_inp = jax.random.normal(V_inp_rng, (4, 30, 2))  # Batch size 4, input size (V) 100, dim_input 2
    S_inp = jax.random.normal(S_inp_rng, (4, 30))  # Batch size 4, input size (V) 100
    neg_S_inp = jax.random.normal(neg_S_inp_rng, (4, 30))  # Batch size 4, input size (V) 100
    rec_net_inp = jax.random.normal(rec_net_inp_rng, (4, 1))  # Batch size 4, input size (V) 100
    # MOONS_CONFIG = {'data_name': 'moons', 'v_size': 100, 's_size': 10, 'batch_size': 128}
    # x = random.normal(key1, (10,))  # Dummy input data
    # params = model.init(key2, x)  # Initialization call
    # jax.tree_util.tree_map(lambda x: x.shape, params)  # Checking output shapes
    mySetModel = SetFunction(params=params)
    # print(mySetModel)
    new_params = mySetModel.init(init_rng, V_inp, S_inp, neg_S_inp)
    # print(new_params)

    # for v_size in range(1, 17):
    #     V_inp = jax.random.normal(V_inp_rng, (1, v_size, 2))  # Batch size 4, input size (V) 100, dim_input 2
    #     powerset = mySetModel.get_powerset(V_inp)
    #     # print(powerset)
    #     bs, vs = V_inp.shape[:2]
    #     q = .5 * jnp.ones((bs, vs))
    #     multilin = mySetModel.apply(new_params, V_inp, powerset, q, method="multilinear_relaxation")
    #     # print(multilin)
    #     # q_i, q_not_i = mySetModel.get_q_i(q)
    #     # true_multilin_grad = mySetModel.apply(new_params, V_inp, q_i, q_not_i, method="true_grad")
    #     # print(true_multilin_grad)
    #     errs = []
    #     for num_samples in [1, 10, 100, 1000, 10000]:
    #         subset_i, subset_not_i, subset_mat = MC_sampling(q, num_samples)
    #         # print(subset_mat)
    #         F_S = mySetModel.apply(new_params, V_inp, subset_mat, fpi=True,  method="F_S")
    #         F_S = F_S.squeeze(-1).mean(1).mean(-1)
    #         # print(F_S)
    #         err = jnp.linalg.norm(multilin - F_S)
    #         print(err)
    #         errs.append(err)
    #     plt.figure()
    #     plt.xscale("log")
    #     plt.plot([1, 10, 100, 1000, 10000], errs)
    #     plt.xlabel(f'# of samples (log)')
    #     plt.ylabel(r'$|\tilde{F} (\boldsymbol{\psi}, \boldsymbol{\theta}) - '
    #                r'\hat{\tilde{F}} (\boldsymbol{\psi}, \boldsymbol{\theta})|$')
    #     plt.title(r'Difference between the true multilinear relaxation and its estimation')
    #     plt.show()
    #     plot_path = f'plots/tests/multilin/V_size_{v_size}'
    #     while os.path.isfile(f'{plot_path}.png'):
    #         plot_path += '+'
    #     plt.savefig(f'{plot_path}.png', bbox_inches="tight")
    #     plt.close()

    layer = SigmoidFixedPointLayer(set_func=mySetModel, samp_func=MC_sampling, num_samples=params['num_samples'])
    rng, X_rng = jax.random.split(rng, 2)
    variables = layer.init(jax.random.key(0), jnp.ones((4, 30)), V_inp)
    all_iterations = []
    for X_rng in range(40):
        X = jax.random.normal(jax.random.key(X_rng), (4, 30))
        Z, iterations, err, errs = layer.apply(variables, X, V_inp)
        print(f"Terminated after {iterations} iterations with error {err}")
        all_iterations.append(iterations)

        plt.figure()
        plt.yscale("log")
        plt.plot(range(iterations), errs)
        plt.xlabel(f'fixed point iterations for X_rng={X_rng}')
        plt.ylabel(r'$|q - q_{next}|$')
        plt.title(r'Difference between fixed point iterations')
        plt.show()
        plot_path = f'plots/early_stop/test_{X_rng}'
        while os.path.isfile(f'{plot_path}.png'):
            plot_path += '+'
        plt.savefig(f'{plot_path}.png', bbox_inches="tight")
        plt.close()

    print(
        f"On average, fixed-point iterations are terminated after {sum(all_iterations) / len(all_iterations):.2f} iterations.")
    # subset_i, subset_not_i = mySet.MC_sampling(q, 500)