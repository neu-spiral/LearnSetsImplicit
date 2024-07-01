import torch
import datetime
import torch.nn as nn
import torch.nn.functional as F

from model.acnn import ACNN
from utils.config import ACNN_CONFIG
from model.celebaCNN import celebaCNN
from model.deepDTA import DeepDTA_Encoder
from utils.pytorch_helper import FF, normal_cdf


class SetFunction(nn.Module):  # nn.Module is the base class for all NN modules. Any model should subclass this class.
    """
        Definition of the set function (F_θ) using a NN.
    """
    def __init__(self, params):
        super(SetFunction, self).__init__()
        self.params = params
        self.dim_feature = 256  # dimension of the NN layers

        self.init_layer = self.define_init_layer()  # custom init layers for different setups
        self.ff = FF(self.dim_feature, 500, 1, self.params.num_layers)  # forward fold?

    def define_init_layer(self):
        """
        Returns the initial layer custom to different setups.
        :return: InitLayer
        """
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

    def MC_sampling(self, q, M):  # we should be able to get rid of this step altogether
        """
        Bernoulli sampling using q as parameters.
        Args:
            q: parameter of Bernoulli distribution (ψ in the paper)
            M: number of samples (m in the paper)

        Returns:
            Sampled subsets F(S+i), F(S)

        """
        bs, vs = q.shape

        q = q.reshape(bs, 1, 1, vs).expand(bs, M, vs, vs)
        sample_matrix = torch.bernoulli(q)

        mask = torch.cat([torch.eye(vs, vs).unsqueeze(0) for _ in range(M)], dim=0).unsqueeze(0).to(q.device)
        # print(f"shape of mask is {mask.shape}")
        matrix_0 = sample_matrix * (1 - mask)
        matrix_1 = matrix_0 + mask
        return matrix_1, matrix_0  # F([x]_+i), F([x]_- i)

    def mean_field_iteration(self, V, subset_i, subset_not_i):  # ψ_i in the paper
        F_1 = self.F_S(V, subset_i, fpi=True).squeeze(-1)
        F_0 = self.F_S(V, subset_not_i, fpi=True).squeeze(-1)
        q = torch.sigmoid((F_1 - F_0).mean(1))
        return q

    def cross_entropy(self, q, S, neg_S):  # Eq. (5) in the paper
        # print(f"shape of S is {S.shape}")
        # print(f"shape of q is {q.shape}")
        loss = - torch.sum((S * torch.log(q + 1e-12) + (1 - S) * torch.log(1 - q + 1e-12)) * neg_S, dim=-1)
        return loss.mean()

    def forward(self, V, S, neg_S, rec_net):  # return cross-entropy loss
        if self.params.mode == 'diffMF':
            # print(f"V type: {type(V)}")
            # print(neg_S.shape)
            if self.params.data_name == 'bindingdb':
                bs = self.params.batch_size
                vs = self.params.v_size
                q = .5 * torch.ones(bs, vs).to(S.device)  # ψ_0 <-- 0.5 * vector(1)
            else:
                bs, vs = V.shape[:2]
                if self.params.data_name == 'celeba':
                    bs = int(bs / 8)
                    vs = self.params.v_size
                q = .5 * torch.ones(bs, vs).to(V.device)  # ψ_0 <-- 0.5 * vector(1)

        else:
            # mode == 'ind' or 'copula'
            q = rec_net.get_vardist(V, S.shape[0]).detach()  # notice the detach here

        for i in range(self.params.RNN_steps):  # MFVI K times where K = RNN_steps
            sample_matrix_1, sample_matrix_0 = self.MC_sampling(q, self.params.num_samples)
            q = self.mean_field_iteration(V, sample_matrix_1, sample_matrix_0)  # ψ

        # there should be an alternative q definition here using IFT and AutoGrad

        loss = self.cross_entropy(q, S, neg_S)
        return loss

    def F_S(self, V, subset_mat, fpi=False):
        # print(V.shape)
        # print(f"shape of V is {V.shape}")
        # print(f"shape of the init layer is {self.init_layer(V).shape}")
        # print(self.init_layer(V).shape)
        # print(f"length of V: {len(V)}")  # length of V: 2
        # print(f"type of V: {type(V)}")  # type of V: <class 'list'>
        # print(f"V[0] shape: {V[0].shape}")  # V[0] shape: torch.Size([1200, 41, 100])
        # print(f"V[1] shape: {V[1].shape}")  # V[1] shape: torch.Size([1200, 20, 1000])
        # print(f"shape of the init layer is {self.init_layer(V).shape}")
        if fpi:
            # to fix point iteration (aka mean-field iteration)
            fea = self.init_layer(V).reshape(subset_mat.shape[0], 1, -1, self.dim_feature)
        else:
            # to encode variational dist
            fea = self.init_layer(V).reshape(subset_mat.shape[0], -1, self.dim_feature)
        # print(f"subset_mat shape {subset_mat.shape}")  # (bs, M, vs, vs)
        # print(f"fea shape {fea.shape}")  # (bs, 1, vs, dim_feature)
        fea = subset_mat @ fea
        fea = self.ff(fea)
        return fea


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
