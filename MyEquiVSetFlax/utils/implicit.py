import jax
import flax
import flax.linen as nn
import jax.numpy as jnp
from functools import partial
from jaxopt import AndersonAcceleration, FixedPointIteration
from jaxopt.linear_solve import solve_gmres, solve_normal_cg
# from model.set_functions_flax import SetFunction, MC_sampling
from typing import Any, Callable


class SigmoidFixedPointLayer(nn.Module):
    fixed_point: Callable
    tol: float = 1e-3
    max_iter: int = 20
    is_test: bool = False
    early_stop: bool = False

    def __call__(self, q_init, V, **kwargs):
        q = q_init
        iterations = 0
        if self.is_test:
            errs = []
        last_err = float('inf')

        # iterate until convergence
        while iterations < self.max_iter:
            q_next = self.fixed_point(V, q)
            err = jnp.linalg.norm(q - q_next)
            if self.is_test:
                errs.append(err)
            q = q_next
            iterations += 1
            if err < self.tol or (self.early_stop and last_err < err):
                break
            last_err = err

        return (q, iterations, err, errs) if self.is_test else (q, iterations, err)


class SigmoidImplicitLayer(nn.Module):  # JAXOpt solves this for us using implicit differentiation
    set_func: Callable
    fixed_point_solver: Any  # AndersonAcceleration or FixedPointIteration
    num_samples: int = 5

    def __call__(self, x, V, **kwargs):
        q = x

        key = jax.random.key(42)
        V_dummy = jnp.ones(shape=V.shape)
        init_params = self.set_func.init(key, V_dummy, q, method='mean_field_iteration')

        # # shape of a single example
        # init = lambda rng, x: self.set_func.init(rng, x[0], x[0], x[0], method='grad_F_S')
        # init_params = self.param("init_params", init, x)

        def set_func_apply(q, V, init_params):
            return self.set_func.apply(init_params, V, q, method='mean_field_iteration')

        solver = self.fixed_point_solver(fixed_point_fun=set_func_apply)

        def batch_run(q, init_params):
            # print(solver.run(V_dummy, q, init_params).params)
            return solver.run(q, V, init_params)

        # We use vmap since we want to compute the fixed point separately for each
        # example in the batch.
        return batch_run(q, init_params).params  # jax.vmap(batch_run, in_axes=(0, None), out_axes=(0, None))(q, init_params)


