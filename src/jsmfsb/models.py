# models.py
# some pre-defined models

import jax.numpy as jnp
from jsmfsb import Spn


def bd(th=[1, 1.1]):
    """Create a birth-death model

    Create and return a Spn object representing a discrete stochastic
    birth-death model.

    Parameters
    ----------
    th: array
        array of length 2 containing the birth and death rates

    Returns
    -------
    Spn model object with rates `th`

    Examples
    --------
    >>> import jsmfsb
    >>> import jax
    >>> bd = jsmfsb.models.bd()
    >>> step = bd.step_gillespie()
    >>> k = jax.random.key(42)
    >>> jsmfsb.sim_time_series(k, bd.m, 0, 50, 0.1, step)
    """
    return Spn(
        ["X"],
        ["Birth", "Death"],
        [[1], [1]],
        [[2], [0]],
        lambda x, t: jnp.array([th[0] * x[0], th[1] * x[0]]),
        [100],
    )


def dimer(th=[0.00166, 0.2]):
    """Create a dimerisation kinetics model

    Create and return a Spn object representing a discrete stochastic
    dimerisation kinetics model.

    Parameters
    ----------
    th: array
        array of length 2 containing the rates of the bind and unbind reactions

    Returns
    -------
    Spn model object with rates `th`

    Examples
    --------
    >>> import jsmfsb
    >>> import jax
    >>> dimer = jsmfsb.models.dimer()
    >>> step = dimer.step_gillespie()
    >>> k = jax.random.key(42)
    >>> jsmfsb.sim_time_series(k, dimer.m, 0, 50, 0.1, step)
    """
    return Spn(
        ["P", "P2"],
        ["Dim", "Diss"],
        [[2, 0], [0, 1]],
        [[0, 1], [2, 0]],
        lambda x, t: jnp.array([th[0] * x[0] * (x[0] - 1) / 2, th[1] * x[1]]),
        [301, 0],
    )


def id(th=[1, 0.1]):
    """Create an immigration-death model

    Create and return a Spn object representing a discrete stochastic
    immigration-death model.

    Parameters
    ----------
    th: array
        array of length 2 containing the immigration and death rates

    Returns
    -------
    Spn model object with rates `th`

    Examples
    --------
    >>> import smfsb
    >>> import jax
    >>> id = jsmfsb.models.id()
    >>> step = id.step_gillespie()
    >>> k = jax.random.key(42)
    >>> jsmfsb.sim_time_series(k, id.m, 0, 50, 0.1, step)
    """
    return Spn(
        ["X"],
        ["Immigration", "Death"],
        [[0], [1]],
        [[1], [0]],
        lambda x, t: jnp.array([th[0], th[1] * x[0]]),
        [0],
    )


def lv(th=[1, 0.005, 0.6]):
    """Create a Lotka-Volterra model

    Create and return a Spn object representing a discrete stochastic
    Lotka-Volterra model.

    Parameters
    ----------
    th: array
        array of length 3 containing the rates of the three governing reactions,
        prey reproduction, predator-prey interaction, and predator death

    Returns
    -------
    Spn model object with rates `th`

    Examples
    --------
    >>> import jsmfsb
    >>> import jax
    >>> lv = jsmfsb.models.lv()
    >>> step = lv.step_gillespie()
    >>> k = jax.random.key(42)
    >>> jsmfsb.sim_time_series(k, lv.m, 0, 50, 0.1, step)
    """
    return Spn(
        ["Prey", "Predator"],
        ["Prey rep", "Inter", "Pred death"],
        [[1, 0], [1, 1], [0, 1]],
        [[2, 0], [0, 2], [0, 0]],
        lambda x, t: jnp.array([th[0] * x[0], th[1] * x[0] * x[1], th[2] * x[1]]),
        [50, 100],
    )


def mm(th=[0.00166, 1e-4, 0.1]):
    """Create a Michaelis-Menten enzyme kinetic model

    Create and return a Spn object representing a discrete stochastic
    Michaelis-Menten enzyme kinetic model.

    Parameters
    ----------
    th: array
        array of length 3 containing the binding, unbinding and production rates

    Returns
    -------
    Spn model object with rates `th`

    Examples
    --------
    >>> import jsmfsb
    >>> import jax
    >>> mm = jsmfsb.models.mm()
    >>> step = mm.step_gillespie()
    >>> k = jax.random.key(42)
    >>> jsmfsb.sim_time_series(k, mm.m, 0, 50, 0.1, step)
    """
    return Spn(
        ["S", "E", "SE", "P"],
        ["Bind", "Unbind", "Produce"],
        [[1, 1, 0, 0], [0, 0, 1, 0], [0, 0, 1, 0]],
        [[0, 0, 1, 0], [1, 1, 0, 0], [0, 1, 0, 1]],
        lambda x, t: jnp.array([th[0] * x[0] * x[1], th[1] * x[2], th[2] * x[2]]),
        [301, 120, 0, 0],
    )


def sir(th=[0.0015, 0.1]):
    """Create a basic SIR compartmental epidemic model

    Create and return a Spn object representing a discrete stochastic
    SIR model.

    Parameters
    ----------
    th: array
        array of length 2 containing the rates of the two governing transitions

    Returns
    -------
    Spn model object with rates `th`

    Examples
    --------
    >>> import jsmfsb
    >>> import jax
    >>> sir = jsmfsb.models.sir()
    >>> step = sir.step_gillespie()
    >>> k = jax.random.key(42)
    >>> jsmfsb.sim_time_series(k, sir.m, 0, 50, 0.1, step)
    """
    return Spn(
        ["S", "I", "R"],
        ["S->I", "I->R"],
        [[1, 1, 0], [0, 1, 0]],
        [[0, 2, 0], [0, 0, 1]],
        lambda x, t: jnp.array([th[0] * x[0] * x[1], th[1] * x[1]]),
        [197, 3, 0],
    )


# eof
