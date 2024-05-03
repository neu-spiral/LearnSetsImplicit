import jax
import jax.numpy as jnp
from jax import random
from jax.lib import xla_bridge
from jax.nn import sigmoid
from functools import partial


def fwd_solver(f, psi_init):
    psi_prev, psi = psi_init, f(psi_init)
    while jnp.linalg.norm(psi_prev - psi) > 1e-5:
        psi_prev, psi = psi, f(psi)
    return psi


def newton_solver(f, psi_init):
    f_root = lambda psi: f(psi) - psi
    g = lambda psi: psi - jnp.linalg.solve(jax.jacobian(f_root)(psi), f_root(psi))
    return fwd_solver(g, psi_init)


def anderson_solver(f, psi_init, m=5, lam=1e-4, max_iter=50, tol=1e-5, beta=1.0):
    x0 = psi_init
    x1 = f(x0)
    x2 = f(x1)
    X = jnp.concatenate([jnp.stack([x0, x1]), jnp.zeros((m - 2, *jnp.shape(x0)))])
    F = jnp.concatenate([jnp.stack([x1, x2]), jnp.zeros((m - 2, *jnp.shape(x0)))])

    res = []
    for k in range(2, max_iter):
        n = min(k, m)
        G = F[:n] - X[:n]
        GTG = jnp.tensordot(G, G, [list(range(1, G.ndim))] * 2)
        H = jnp.block([[jnp.zeros((1, 1)), jnp.ones((1, n))],
                       [jnp.ones((n, 1)), GTG]]) + lam * jnp.eye(n + 1)
        alpha = jnp.linalg.solve(H, jnp.zeros(n + 1).at[0].set(1))[1:]

        xk = beta * jnp.dot(alpha, F[:n]) + (1 - beta) * jnp.dot(alpha, X[:n])
        X = X.at[k % m].set(xk)
        F = F.at[k % m].set(f(xk))

        res = jnp.linalg.norm(F[k % m] - X[k % m]) / (1e-5 + jnp.linalg.norm(F[k % m]))
        if res < tol:
            break
    return xk


@partial(jax.custom_vjp, nondiff_argnums=(0, 1))
def fixed_point_layer(solver, f, params, x):
    psi_star = solver(lambda psi: f(params, x, psi), psi_init=jnp.zeros_like(x))
    return psi_star


def fixed_point_layer_fwd(solver, f, params, x):
    psi_star = fixed_point_layer(solver, f, params, x)
    return psi_star, (params, x, psi_star)


def fixed_point_layer_bwd(solver, f, res, psi_star_bar):
    params, x, psi_star = res
    _, vjp_a = jax.vjp(lambda params, x: f(params, x, psi_star), params, x)
    _, vjp_psi = jax.vjp(lambda psi: f(params, x, psi), psi_star)
    return vjp_a(solver(lambda u: vjp_psi(u)[0] + psi_star_bar, psi_init=jnp.zeros_like(psi_star)))


fixed_point_layer.defvjp(fixed_point_layer_fwd, fixed_point_layer_bwd)

if __name__ == "__main__":
    # if the next statement prints 'gpu' it means we are actually using the gpu.
    print(xla_bridge.get_backend().platform)

    # scalar valued f
    # f = lambda theta, x, psi: sigmoid(jnp.dot(theta, psi) + x)

    # vector valued f
    def f(x):
        return sigmoid(x) # sigmoid function returns an array
    ndim = 10
    theta = random.normal(random.PRNGKey(0), (ndim, ndim)) / jnp.sqrt(ndim)
    x = random.normal(random.PRNGKey(1), (ndim,))

    psi_star = fixed_point_layer(fwd_solver, f, theta, x)
    print(psi_star)
    psi_star = fixed_point_layer(newton_solver, f, theta, x)
    print(psi_star)
    psi_star = fixed_point_layer(anderson_solver, f, theta, x)
    print(psi_star)

    g = jax.grad(lambda theta: fixed_point_layer(fwd_solver, f, theta, x).sum())(theta)
    print(g[0])
    g = jax.grad(lambda theta: fixed_point_layer(newton_solver, f, theta, x).sum())(theta)
    print(g[0])
    g = jax.grad(lambda theta: fixed_point_layer(anderson_solver, f, theta, x).sum())(theta)
    print(g[0])

    def f(x):
        return jnp.sin(x) * x ** 2

    x = 2.
    y = f(x)
    print(y)

    delta_x = 1.
    y, delta_y = jax.jvp(f, (x,), (delta_x,))
    print(y)
    print(delta_y)

    eps = 1e-4
    delta_y_approx = (f(x + eps * v) - f(x)) / eps
    print(delta_y_approx)
