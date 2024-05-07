import jax.numpy as jnp
from jaxopt import FixedPointIteration


def T(x, theta):  # contractive map
    return 0.5 * x + theta


fpi = FixedPointIteration(fixed_point_fun=T)
x_init = jnp.array(0.)
theta = jnp.array(0.5)
print(fpi.run(x_init, theta).params)
