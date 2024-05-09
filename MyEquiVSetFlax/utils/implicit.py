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

    @nn.compact
    def __call__(self, q_init, V, **kwargs):
        q = q_init
        init = lambda rng, q: self.fixed_point.init(rng, q, V)
        block_params = self.param("block_params", init, q)
        iterations = 0
        if self.is_test:
            errs = []
        last_err = float('inf')

        # iterate until convergence
        while iterations < self.max_iter:
            q_next = self.fixed_point.apply(block_params, q, V)
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
    fixed_point: Callable
    fixed_point_solver: Any  # AndersonAcceleration or FixedPointIteration

    # @partial(nn.transforms.vmap,
    #          variable_axes={'params': None},
    #          split_rngs={'params': False},
    #          in_axes=(None, 0))
    @nn.compact
    def __call__(self, q_init, V, **kwargs):
        q = q_init
        init = lambda rng, q: self.fixed_point.init(rng, q, V)
        block_params = self.param("block_params", init, q)

        def set_func_apply(q, V):
            return self.fixed_point.apply(block_params, q, V)

        solver = self.fixed_point_solver(fixed_point_fun=set_func_apply)

        def batch_run(q, V, block_params):
            # print(solver.run(V_dummy, q, init_params).params)
            return solver.run(q, V, block_params)

        # We use vmap since we want to compute the fixed point separately for each
        # example in the batch.
        return jax.vmap(batch_run, in_axes=(0, None), out_axes=0)(q, V, block_params)  # understand what vmap does and modify this line
