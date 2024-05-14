import jax
import flax
import flax.linen as nn
import jax.numpy as jnp
from functools import partial
from jax import jit
from jax import lax
from jaxopt import AndersonAcceleration, FixedPointIteration
from jaxopt.linear_solve import solve_gmres, solve_normal_cg
# from model.set_functions_flax import SetFunction, MC_sampling
from typing import Any, Callable


class SigmoidFixedPointLayer(nn.Module):
    mfvi: Callable  # nn.Module
    tol: float = 1e-3
    max_iter: int = 20
    is_test: bool = False
    early_stop: bool = False

    @nn.compact
    def __call__(self, q_init, V, **kwargs):
        # q = q_init
        # iterations = 0
        # if self.is_test:
        #     errs = []
        # last_err = float('inf')
        #
        # # iterate until convergence
        # while iterations < self.max_iter:
        #     q_next = self.fixed_point(q, V)
        #     err = jnp.linalg.norm(q - q_next)
        #     if self.is_test:
        #         errs.append(err)
        #     q = q_next
        #     iterations += 1
        #     if jnp.where(err < self.tol) or (self.early_stop and jnp.where(last_err < err)):
        #         break
        #     last_err = err

        init = lambda rng, q, V: self.mfvi.init(rng, q, V)
        mfvi_params = self.param("mfvi_params", init, q_init, V)

        # key = jax.random.PRNGKey(0)
        # params = self.mfvi.init(key, q_init, V)

        # @partial(jit, static_argnames=['err'])
        def body(mdl, carry):
            q, iterations, last_err = carry
            q_next = mdl.mfvi.apply(mfvi_params, q, V)
            err = jnp.linalg.norm(q - q_next)
            return q_next, iterations + 1, err

        # @partial(jit, static_argnames=['err'])
        def cond(mdl, carry):
            q, iterations, err = carry
            # break_condition = (err > 1e-3)
            return (err > mdl.tol) & (iterations < mdl.max_iter)
            # return (iterations < mdl.max_iter)

        # return lax.while_loop(cond, body, (q_init, 0, float('inf')))
        return nn.while_loop(cond, body, mdl=self, init=(q_init, 0, float('inf')))
        # return (q, iterations, err, errs) if self.is_test else (q, iterations, err)
        # return q


class SigmoidImplicitLayer(nn.Module):  # JAXOpt solves this for us using implicit differentiation
    mfvi: Callable
    fixed_point_solver: Any  # AndersonAcceleration or FixedPointIteration

    # def setup(self):
    #     self.init = lambda rng, q, V: self.mfvi.init(rng, q, V)
        # self.mfvi_params = self.param("mfvi_params", self.init, q_init, V)

    # @partial(nn.transforms.vmap,
    #          variable_axes={'params': None},
    #          split_rngs={'params': False},
    #          in_axes=(None, 0))
    # @partial(
    #     nn.transforms.scan,
    #     variable_broadcast='params',
    #     split_rngs={'params': False})
    @nn.compact
    def __call__(self, q_init, V, **kwargs):
        # q = q_init
        init = lambda rng, q, V: self.mfvi.init(rng, q, V)
        # # if self.is_initializing():
        mfvi_params = self.param("mfvi_params", init, q_init, V)
        # # mfvi_params = self.mfvi.init(jax.random.key(42), q_init, V)

        def mfvi_apply(q, V, mfvi_params):
            q, mfvi_params = self.mfvi.apply(mfvi_params, q, V, mutable=True)
            return q
        # def mfvi_apply(q, V):
        #     q = self.mfvi(q, V)
        #     return q

        solver = self.fixed_point_solver(fixed_point_fun=mfvi_apply)

        def batch_run(q, V, mfvi_params):
            # print(solver.run(V_dummy, q, init_params).params)
            return solver.run(q, V, mfvi_params)[0]

        # def batch_run(q, V):
        #     # print(solver.run(V_dummy, q, init_params).params)
        #     return solver.run(q, V)[0]

        # We use vmap since we want to compute the fixed point separately for each
        # example in the batch.
        # return jax.vmap(batch_run, in_axes=(0, 0, None), out_axes=0)(q_init, V, mfvi_params)  # understand what vmap does and modify this line
        return batch_run(q_init, V, mfvi_params)
        # return batch_run(q_init, V)


# class SigmoidImplicitLayer(nn.Module):  # JAXOpt solves this for us using implicit differentiation
#     mfvi: Callable
#     fixed_point_solver: Any  # AndersonAcceleration or FixedPointIteration
#
#     # def setup(self):
#     #     self.init = lambda rng, q, V: self.mfvi.init(rng, q, V)
#         # self.mfvi_params = self.param("mfvi_params", self.init, q_init, V)
#
#     # @partial(nn.transforms.vmap,
#     #          variable_axes={'params': None},
#     #          split_rngs={'params': False},
#     #          in_axes=(None, 0))
#     @nn.compact
#     def __call__(self, q_init, V, **kwargs):
#         # q = q_init
#         # init = lambda rng, q, V: self.mfvi.init(rng, q, V)
#         # if self.is_initializing():
#         # mfvi_params = self.param("mfvi_params", self.init, q_init, V)
#
#         def mfvi_apply(q, V):
#             return self.mfvi(q, V)
#
#         solver = self.fixed_point_solver(fixed_point_fun=mfvi_apply)
#
#         def batch_run(q, V):
#             # print(solver.run(V_dummy, q, init_params).params)
#             return solver.run(q, V).params
#
#         # We use vmap since we want to compute the fixed point separately for each
#         # example in the batch.
#         return jax.vmap(batch_run, in_axes=(None, 1), out_axes=0)(q_init, V)  # understand what vmap does and modify this line
#         # return batch_run(q_init, V)