import sys

# setting path
sys.path.append('../EquiVSetFlax')
import jax
import datetime
import flax
import flax.linen as nn
import jax.numpy as jnp
import numpy as np
from utils.flax_helper import FF, normal_cdf


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
    q = jnp.broadcast_to(q.reshape(bs, 1, 1, vs), (bs, M, vs, vs))  # .expand(bs, M, vs, vs)
    # bs = the batch size
    q = jax.device_put(q)
    sample_matrix = jax.random.bernoulli(jax.random.PRNGKey(758493), q)  # we should have torch uniform outside

    mask = jnp.expand_dims(jnp.concatenate([jnp.expand_dims(jnp.eye(vs, vs), axis=0) for _ in range(M)], axis=0),
                           axis=0)
    # what does this line do?
    # after the first unsqueeze we have a 3D tensor with 1 channel, vs rows, and vs columns
    # after concat we have a 3D tensor with M channels, vs rows, and vs columns
    # after the second unsqueeze we have a (1, M, vs, vs) shaped tensor
    matrix_0 = sample_matrix * (1 - mask)  # element_wise multiplication
    matrix_1 = matrix_0 + mask
    return matrix_1, matrix_0  # F([x]_+i), F([x]_- i)


# noinspection PyAttributeOutsideInit
class SetFunction(nn.Module):  # nn.Module is the base class for all NN modules. Any model should subclass this class.
    """
        Definition of the set function (F_θ) using a NN.
    """
    params: dict
    dim_feature: int = 256

    # params = {v_size: 30,
    #           s_size: 10,
    #           num_layers: 2,
    #           batch_size: 4,
    #           lr: 0.0001,
    #           weight_decay: 1e-5,
    #           init: 0.05,
    #           clip: 10,
    #           epochs: 100,
    #           num_runs: 1,
    #           num_bad_epochs: 6,
    #           num_workers: 2
    #           }

    # For instance, if we define more functions on a module besides __call__ and want to reuse some modules, it is
    # recommended to use the setup version.
    def setup(self):
        self.init_layer = self.define_init_layer()
        self.ff = FF(self.dim_feature, 500, 1, self.params['num_layers'])

    def define_init_layer(self):
        """
        Returns the initial layer custom to different setups.
        :return: InitLayer
        """
        return nn.Dense(features=self.dim_feature)

    def mean_field_iteration(self, V, subset_i, subset_not_i):  # ψ_i in the paper
        q = jax.nn.sigmoid((self.grad_F_S(V, subset_i, subset_not_i)).mean(1))
        return q

    def __call__(self, V, S, neg_S, **kwargs):
        """"returns cross-entropy loss."""
        bs, vs = V.shape[:2]
        q = .5 * jnp.ones((bs, vs))  # ψ_0 <-- 0.5 * vector(1)
        # q = jax.random.uniform(jax.random.PRNGKey(758493), shape=(bs, vs))

        for i in range(self.params['RNN_steps']):  # MFVI K times where K = RNN_steps
            sample_matrix_1, sample_matrix_0 = MC_sampling(q, self.params['num_samples'])
            q = self.mean_field_iteration(V, sample_matrix_1, sample_matrix_0)  # ψ

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
        fea = self.ff(fea)  # goes thru FF block
        # self.ff.apply(params, fea)
        return fea

    # def multilinear_relaxation(F_S, y):
    #     out = 0.0
    #     for i in range(2 ** len(y)):
    #         binary_vector = map(int, list(bin(i)[2:]))
    #         if len(binary_vector) < len(y):
    #             binary_vector = [0] * (len(y) - len(binary_vector)) + binary_vector
    #         x = dict(zip(y.iterkeys(), binary_vector))
    #         new_term = F_S(x)
    #         for key in y:
    #             if x[key] == 0:
    #                 new_term *= (1.0 - y[key])
    #             else:
    #                 new_term *= y[key]
    #         out += new_term
    #     return out

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
