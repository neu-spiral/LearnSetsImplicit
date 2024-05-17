import flax.linen as nn
import optax
from model.base_model_flax import Base_Model
from model.set_functions_flax import SetFunction, RecNet
from typing import Any, Dict
# import model.train_module


# class EquiVSetTrainer(TrainerModule):
#     def __init__(self, hparams):
#         super().__init__(hparams=hparams)
#
#     def define_parameters(self):  # def setup(self): might look more JAXian
#         self.set_func = SetFunction(params=self.hparams)
#         self.rec_net = RecNet(params=self.hparams) if self.hparams.mode != 'diffMF' else None
#
#     def configure_optimizers(self):
#         optim_energy = flax.optim.Adam(self.set_func.parameters(), lr=self.hparams.lr,
#                                        weight_decay=self.hparams.weight_decay)
#         optim_var = flax.optim.Adam(self.rec_net.parameters(), lr=self.hparams.lr,
#                                     weight_decay=self.hparams.weight_decay) if self.hparams.mode != 'diffMF' else None
#         return optim_energy, optim_var
#
#     def configure_gradient_clippers(self):
#         return [(self.parameters(), self.hparams.clip)]
#
#     def inference(self, V, bs):
#         if self.hparams.mode == 'diffMF':
#             bs, vs = V.shape[:2]
#             q = .5 * torch.ones(bs, vs).to(V.device)
#         else:
#             # mode == 'ind' or 'copula'
#             q = self.rec_net.get_vardist(V, bs)
#
#         for i in range(self.hparams.RNN_steps):
#             sample_matrix_1, sample_matrix_0 = self.set_func.MC_sampling(q, self.hparams.num_samples)
#             q = self.set_func.mean_field_iteration(V, sample_matrix_1, sample_matrix_0)
#
#         return q
#
#     def get_hparams_grid(self):
#         grid = Base_Model.get_general_hparams_grid()
#         grid.update({
#             'RNN_steps': [1],
#             'num_samples': [1, 5, 10, 15, 20],
#             'rank': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
#             'tau': [0.01, 0.03, 0.05, 0.07, 0.09, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 5, 10],
#         })
#         return grid
#
#     @staticmethod
#     def get_model_specific_argparser():
#         parser = Base_Model.get_general_argparser()
#
#         parser.add_argument('--mode', type=str, default='diffMF',
#                             choices=['diffMF', 'ind', 'copula'],
#                             help='name of the variant model [%(default)s]')
#         parser.add_argument('--RNN_steps', type=int, default=1,
#                             help='num of RNN steps [%(default)d], K in the paper')
#         parser.add_argument('--num_samples', type=int, default=5,
#                             help='num of Monte Carlo samples [%(default)d]')
#         parser.add_argument('--rank', type=int, default=5,
#                             help='rank of the perturbation matrix [%(default)d]')
#         parser.add_argument('--tau', type=float, default=0.1,
#                             help='temperature of the relaxed multivariate bernoulli [%(default)g]')
#         parser.add_argument('--neg_num', type=int, default=1,
#                             help='num of the negative item [%(default)d]')
#
#         return parser


class EquiVSet(Base_Model):
    def __init__(self, hparams: Dict[str, Any]):
        super().__init__(hparams=hparams)  # might include a model_class

    def setup(self):  # define_parameters() is renamed as setup
        # these are okay for now, they will need initialization later
        self.set_func = SetFunction(params=self.hparams)
        self.rec_net = RecNet(params=self.hparams) if self.hparams.mode != 'diffMF' else None

    # In general you shouldn’t call .setup() yourself, if you need to get access to a field or submodule defined inside
    # setup you can instead create a function to extract it and pass it to nn.apply:
    def get_parameters(self):
        set_func = self.set_func.clone()
        rec_net = self.rec_net.clone()
        return set_func, recnet

    def configure_optimizers(self):  # might be implemented in the parent class, equivalent of init_optimizer in
        # train_module.py
        # adam doesn't have a weight_decay parameter, adamw has
        optim_energy = optax.adam(learning_rate=self.hparams.lr)
        # optim_energy = optax.adam(self.set_func.parameters(), lr=self.hparams.lr,
        #                           weight_decay=self.hparams.weight_decay)
        optim_var = optax.adam(self.rec_net.parameters(), lr=self.hparams.lr,
                               weight_decay=self.hparams.weight_decay) if self.hparams.mode != 'diffMF' else None
        # this just returns opt_class in train_module.py
        return optim_energy, optim_var

    def configure_gradient_clippers(self):
        return [(self.parameters(), self.hparams.clip)]

    def inference(self, V, bs):
        if self.hparams.mode == 'diffMF':
            bs, vs = V.shape[:2]
            if self.hparams.data_name == 'celeba' or self.hparams.data_name == 'pdbbind':
                bs = int(bs / 8)
                vs = self.hparams.v_size
            q = .5 * torch.ones(bs, vs).to(V.device)
        else:
            # ignoring here for now
            # mode == 'ind' or 'copula'
            q = self.rec_net.get_vardist(V, bs)

        for i in range(self.hparams.RNN_steps):
            sample_matrix_1, sample_matrix_0 = self.set_func.MC_sampling(q, self.hparams.num_samples)
            q = self.set_func.mean_field_iteration(V, sample_matrix_1, sample_matrix_0)

        return q

    def get_hparams_grid(self):
        grid = Base_Model.get_general_hparams_grid()
        grid.update({
            'RNN_steps': [1],
            'num_samples': [1, 5, 10, 15, 20],
            'rank': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
            'tau': [0.01, 0.03, 0.05, 0.07, 0.09, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 5, 10],
        })
        return grid

    @staticmethod
    def get_model_specific_argparser():
        parser = Base_Model.get_general_argparser()

        parser.add_argument('--mode', type=str, default='diffMF',
                            choices=['diffMF', 'ind', 'copula'],
                            help='name of the variant model [%(default)s]')
        parser.add_argument('--RNN_steps', type=int, default=1,
                            help='num of RNN steps [%(default)d], K in the paper')
        parser.add_argument('--num_samples', type=int, default=5,
                            help='num of Monte Carlo samples [%(default)d]')
        parser.add_argument('--rank', type=int, default=5,
                            help='rank of the perturbation matrix [%(default)d]')
        parser.add_argument('--tau', type=float, default=0.1,
                            help='temperature of the relaxed multivariate bernoulli [%(default)g]')
        parser.add_argument('--neg_num', type=int, default=1,
                            help='num of the negative item [%(default)d]')

        return parser
