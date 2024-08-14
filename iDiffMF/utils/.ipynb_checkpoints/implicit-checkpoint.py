import jax
import flax
import flax.linen as nn
import jax.numpy as jnp
from jax import lax
from jaxopt import AndersonAcceleration, FixedPointIteration
from jaxopt.linear_solve import solve_gmres, solve_normal_cg
from typing import Any, Callable


class SigmoidFixedPointLayer(nn.Module):
    mfvi: Callable  # nn.Module
    tol: float = 1e-3
    max_iter: int = 20
    is_test: bool = False
    early_stop: bool = False

    @nn.compact
    def __call__(self, q_init, V, **kwargs):

        init = lambda rng, q, V: self.mfvi.init(rng, q, V)
        mfvi_params = self.param("mfvi_params", init, q_init, V)


        def body(mdl, carry):
            q, iterations, last_err = carry
            q_next = mdl.mfvi.apply(mfvi_params, q, V)
            err = jnp.linalg.norm(q - q_next)
            return q_next, iterations + 1, err

        def cond(mdl, carry):
            q, iterations, err = carry
            # break_condition = (err > 1e-3)
            return (err > mdl.tol) & (iterations < mdl.max_iter)

        return nn.while_loop(cond, body, mdl=self, init=(q_init, 0, float('inf')))



class SigmoidImplicitLayer(nn.Module):  # JAXOpt solves this for us using implicit differentiation
    mfvi: Callable
    fixed_point_solver: Any  # AndersonAcceleration or FixedPointIteration

    @nn.compact
    def __call__(self, q_init, V, **kwargs):
        init = lambda rng, q, V: self.mfvi.init(rng, q, V)
        mfvi_params = self.param("mfvi_params", init, q_init, V)

        def mfvi_apply(q, V, mfvi_params):
            return self.mfvi.apply(mfvi_params, q, V)

        solver = self.fixed_point_solver(fixed_point_fun=mfvi_apply)

        def batch_run(q, V, mfvi_params):
            return solver.run(q, V, mfvi_params)[0]

        return batch_run(q_init, V, mfvi_params)
