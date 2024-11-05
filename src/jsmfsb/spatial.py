# spatial code from chapter 9
# Note that the actual simulation code is in the Spn object in the spn module

import jax
from jax import jit
import jax.numpy as jnp
import jax.lax as jl


def sim_time_series_1d(key, x0, t0, tt, dt, step_fun, verb=False):
    """Simulate a model on a regular grid of times, using a function (closure)
    for advancing the state of the model

    This function simulates single realisation of a model on a 1D
    regular spatial grid and regular grid of times using a function
    (closure) for advancing the state of the model, such as created by
    `step_gillespie_1d`.

    Parameters
    ----------
    key: JAX random number key
      Initial random number key to seed the simulation.
    x0 : array
      The initial state of the process at time `t0`, a matrix with
      rows corresponding to reacting species and columns
      corresponding to spatial location.
    t0 : float
      The initial time to be associated with the initial state `x0`.
    tt : float
      The terminal time of the simulation.
    dt : float
      The time step of the output. Note that this time step relates
      only to the recorded output, and has no bearing on the
      accuracy of the simulation process.
    step_fun : function
      A function (closure) for advancing the state of the process,
      such as produced by `step_gillespie_1d`.
    verb : boolean
      Output progress to the console (this function can be very slow).

    Returns
    -------
    A 3d array representing the simulated process. The dimensions
    are species, space, and time.

    Examples
    --------
    >>> import jsmfsb.models
    >>> import jax
    >>> import jax.numpy as jnp
    >>> lv = jsmfsb.models.lv()
    >>> stepLv1d = lv.step_gillespie_1d(jnp.array([0.6,0.6]))
    >>> N = 10
    >>> T = 5
    >>> x0 = jnp.zeros((2,N))
    >>> x0 = x0.at[:,int(N/2)].set(lv.m)
    >>> k0 = jax.random.key(42)
    >>> jsmfsb.sim_time_series_1d(k0, x0, 0, T, 1, stepLv1d, True)
    """
    nn = int((tt - t0) // dt + 1)
    u, n = x0.shape
    keys = jax.random.split(key, nn)

    @jit
    def advance(state, key):
        x, t = state
        if verb:
            jax.debug.print("{t}", t=t)
        x = step_fun(key, x, t, dt)
        t = t + dt
        return (x, t), x

    _, arr = jl.scan(advance, (x0, t0), keys)
    return jnp.moveaxis(arr, 0, 2)


def sim_time_series_2d(key, x0, t0, tt, dt, step_fun, verb=False):
    """Simulate a model on a regular grid of times, using a function (closure)
    for advancing the state of the model

    This function simulates single realisation of a model on a 2D
    regular spatial grid and regular grid of times using a function
    (closure) for advancing the state of the model, such as created by
    `step_gillespie_2d`.

    Parameters
    ----------
    key: JAX random number key
      Random key to seed the simulation.
    x0 : array
      The initial state of the process at time `t0`, a 3d array with
      dimensions corresponding to reacting species and then two
      corresponding to spatial location.
    t0 : float
      The initial time to be associated with the initial state `x0`.
    tt : float
      The terminal time of the simulation.
    dt : float
      The time step of the output. Note that this time step relates
      only to the recorded output, and has no bearing on the
      accuracy of the simulation process.
    step_fun : function
      A function (closure) for advancing the state of the process,
      such as produced by `step_gillespie_2d`.
    verb : boolean
      Output progress to the console (this function can be very slow).

    Returns
    -------
    A 4d array representing the simulated process. The dimensions
    are species, two space, and time.

    Examples
    --------
    >>> import jsmfsb.models
    >>> import jax
    >>> import jax.numpy as jnp
    >>> lv = jsmfsb.models.lv()
    >>> stepLv2d = lv.step_gillespie_2d(jnp.array([0.6,0.6]))
    >>> M = 10
    >>> N = 15
    >>> T = 5
    >>> x0 = jnp.zeros((2,M,N))
    >>> x0 = x0.at[:,int(M/2),int(N/2)].set(lv.m)
    >>> k0 = jax.random.key(42)
    >>> jsmfsb.sim_time_series_2d(k0, x0, 0, T, 1, stepLv2d, True)
    """
    nn = int((tt - t0) // dt + 1)
    u, m, n = x0.shape
    keys = jax.random.split(key, nn)

    @jit
    def advance(state, key):
        x, t = state
        if verb:
            jax.debug.print("{t}", t=t)
        x = step_fun(key, x, t, dt)
        t = t + dt
        return (x, t), x

    _, arr = jl.scan(advance, (x0, t0), keys)
    return jnp.moveaxis(arr, 0, 3)


# eof
