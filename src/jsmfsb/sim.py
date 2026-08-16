# sim.py

# simulation functions


import jax
from jax import jit
import jax.numpy as jnp
import jax.lax as jl


def sim_time_series(key, x0, t0, tt, dt, step_fun):
    """Simulate a model on a regular grid of times, using a function (closure)
    for advancing the state of the model

    This function simulates single realisation of a model on a regular
    grid of times using a function (closure) for advancing the state
    of the model, such as created by ‘step_gillespie’ or
    ‘step_cle’.

    Parameters
    ----------
    key: JAX random number key
        An unused random number key.
    x0: array of numbers
        The intial state of the system at time t0
    t0: float
        This intial time to be associated with the intial state.
    tt: float
        The terminal time of the simulation.
    dt: float
        The time step of the output. Note that this time step relates only to
        the recorded output, and has no bearing on the accuracy of the simulation
        process.
    step_fun: function
        A function (closure) for advancing the state of the process,
        such as produced by ‘step_gillespie’ or ‘step_cle’.

    Returns
    -------
    A matrix with rows representing the state of the system at successive times.

    Examples
    --------
    >>> import jax
    >>> import jsmfsb.models
    >>> lv = jsmfsb.models.lv()
    >>> stepLv = lv.step_gillespie()
    >>> jsmfsb.sim_time_series(jax.random.key(42), lv.m, 0, 100, 0.1, stepLv)
    """
    n = int((tt - t0) // dt)
    keys = jax.random.split(key, n)

    @jit
    def advance(state, key):
        x, t = state
        x = step_fun(key, x, t, dt)
        t = t + dt
        return (x, t), x

    _, mat = jl.scan(advance, (x0, t0), keys)
    return jnp.insert(mat, 0, x0, 0)


def sim_times(key, x0, t0, times, step_fun):
    """Simulate a model on a provided grid of times, using a function (closure)
    for advancing the state of the model

    This function simulates single realisation of a model on a provided
    grid of times using a function (closure) for advancing the state
    of the model, such as created by ‘step_gillespie’ or
    ‘step_euler’.

    Parameters
    ----------
    key: JAX random number key.
        An unused key.
    x0: array of numbers
        The intial state of the system at time t0
    t0: float
        This intial time to be associated with the intial state.
    times: array of numbers
        A vector of times at which the process is required. It is assumed
        that the times are in increasing order and that the first time is
        at least as big as `t0`.
    step_fun: function
        A function (closure) for advancing the state of the process,
        such as produced by ‘step_gillespie’ or ‘step_euler’.

    Returns
    -------
    A matrix with rows representing the state of the system at the
    required times.

    Examples
    --------
    >>> import jsmfsb.models
    >>> import jax
    >>> import jax.numpy as jnp
    >>> lv = jsmfsb.models.lv()
    >>> stepLv = lv.step_gillespie()
    >>> k0 = jax.random.key(33)
    >>> jsmfsb.sim_times(k0, lv.m, 0, jnp.array([1.0, 2, 5, 10]), stepLv)
    """
    times = jnp.array(times)
    n = len(times)
    u = len(x0)
    mat = jnp.zeros((n, u))
    x = x0
    t = t0
    for i in range(n):  # TODO: replace with a scan
        key, k1 = jax.random.split(key)
        x = step_fun(k1, x, t, times[i] - t)
        t = times[i]
        mat = mat.at[i, :].set(x)
    return mat


def sim_sample(key, n, x0, t0, deltat, step_fun, batch_size=None):
    """Simulate a many realisations of a model at a given fixed time in the
    future given an initial time and state, using a function (closure) for
    advancing the state of the model

    This function simulates many realisations of a model at a given
    fixed time in the future given an initial time and state, using a
    function (closure) for advancing the state of the model , such as
    created by ‘step_gillespie’ or ‘step_cle’.

    Parameters
    ----------
    key: JAX random number key
        An unused random number key.
    n: int
        The number of samples required.
    x0: array of numbers
        The intial state of the system at time t0.
    t0: float
        The intial time to be associated with the initial state.
    deltat: float
        The amount of time in the future of t0 at which samples of the
        system state are required.
    step_fun: function
        A function (closure) for advancing the state of the process,
        such as produced by `step_gillespie' or `step_cle'.
    batch_size: int
        A batch size for "jax.lax.map". If provided, will parallelise.

    Returns
    -------
    A matrix with rows representing simulated states at time t0+deltat.

    Examples
    --------
    >>> import jax
    >>> import jsmfsb.models
    >>> lv = jsmfsb.models.lv()
    >>> stepLv = lv.step_gillespie()
    >>> jsmfsb.sim_sample(jax.random.key(42), 10, lv.m, 0, 30, stepLv)
    """
    keys = jax.random.split(key, n)
    mat = jl.map(lambda k: step_fun(k, x0, t0, deltat), keys, batch_size=batch_size)
    return mat


# eof
