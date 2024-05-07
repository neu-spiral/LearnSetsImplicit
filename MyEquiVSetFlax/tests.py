import jax
import flax
import flax.linen as nn
import jax.numpy as jnp
from functools import partial
from jaxopt import AndersonAcceleration, FixedPointIteration
from jaxopt.linear_solve import solve_gmres, solve_normal_cg
from model.set_functions_flax import SetFunction, MC_sampling
from utils.implicit import SigmoidFixedPointLayer, SigmoidImplicitLayer
import sys

# setting path
sys.path.append('../MyEquiVSetFlax')


def manual_fixedpoint_convergence(fixed_point_layer, V, num_tests=100):
    for X_rng in range(num_tests):
        X = jax.random.normal(jax.random.key(X_rng), (4, 30))
        Z, iterations, err, errs = fixed_point_layer.apply(variables, X, V)
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

    print(f"On average, fixed-point iterations are terminated after {sum(all_iterations) / len(all_iterations):.2f} "
          f"iterations.")


def multilinear_relaxation_estimation(max_v_size=17):
    for v_size in range(1, max_v_size):
        V_inp = jax.random.normal(V_inp_rng, (1, v_size, 2))  # Batch size 4, input size (V) 100, dim_input 2
        powerset = mySetModel.get_powerset(V_inp)
        # print(powerset)
        bs, vs = V_inp.shape[:2]
        q = .5 * jnp.ones((bs, vs))
        multilin = mySetModel.apply(new_params, V_inp, powerset, q, method="multilinear_relaxation")
        # print(multilin)
        # q_i, q_not_i = mySetModel.get_q_i(q)
        # true_multilin_grad = mySetModel.apply(new_params, V_inp, q_i, q_not_i, method="true_grad")
        # print(true_multilin_grad)
        errs = []
        for num_samples in [1, 10, 100, 1000, 10000]:
            subset_i, subset_not_i, subset_mat = MC_sampling(q, num_samples)
            # print(subset_mat)
            F_S = mySetModel.apply(new_params, V_inp, subset_mat, fpi=True,  method="F_S")
            F_S = F_S.squeeze(-1).mean(1).mean(-1)
            # print(F_S)
            err = jnp.linalg.norm(multilin - F_S)
            print(err)
            errs.append(err)
        plt.figure()
        plt.xscale("log")
        plt.plot([1, 10, 100, 1000, 10000], errs)
        plt.xlabel(f'# of samples (log)')
        plt.ylabel(r'$|\tilde{F} (\boldsymbol{\psi}, \boldsymbol{\theta}) - '
                   r'\hat{\tilde{F}} (\boldsymbol{\psi}, \boldsymbol{\theta})|$')
        plt.title(r'Difference between the true multilinear relaxation and its estimation')
        plt.show()
        plot_path = f'plots/tests/multilin/V_size_{v_size}'
        while os.path.isfile(f'{plot_path}.png'):
            plot_path += '+'
        plt.savefig(f'{plot_path}.png', bbox_inches="tight")
        plt.close()


def jaxopt_fixedpoint_sanity_check(manual_layer, jaxopt_layer, variables, V, num_tests=100):
    for X_rng in range(num_tests):
        X = jax.random.normal(jax.random.key(X_rng), (4, 30))
        Z, iterations, err, errs = manual_layer.apply(variables, X, V)
        q_jaxopt = jaxopt_layer.apply(variables, X, V)
        err = jnp.linalg.norm(Z - q_jaxopt)
        assert err < 1e-3, f"The distance between fixed-point solutions are {err}."
    print(f"The distance between the manuel and the JAXOpt implementation of fixed-point solutions is less than "
              f"{1e-3} for {100} starting points.")


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
    # # subset_i, subset_not_i = mySet.MC_sampling(q, 500)
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
    X = jax.random.normal(X_rng, (4, 30))
    z, iterations, err, errs = layer.apply(variables, X, V_inp)
    print(f"Terminated after {iterations} iterations with error {err}.")
    # print(z)
    all_iterations = []

    # manual_fixedpoint_convergence(layer, V_inp)

    implicit_solver = partial(solve_gmres, tol=1e-3, maxiter=100)
    fixed_point_solver = partial(FixedPointIteration,
                                 maxiter=100,
                                 tol=1e-3, implicit_diff=True,
                                 implicit_diff_solve=implicit_solver,
                                 verbose=False)

    implicit_layer = SigmoidImplicitLayer(set_func=mySetModel, fixed_point_solver=fixed_point_solver,
                                          num_samples=params['num_samples'])

    variables = implicit_layer.init(jax.random.key(0), jnp.ones((4, 30)), V_inp)
    q_jaxopt = implicit_layer.apply(variables, X, V_inp)
    print(jnp.linalg.norm(z - q_jaxopt))

    jaxopt_fixedpoint_sanity_check(layer, implicit_layer, variables, V_inp)


